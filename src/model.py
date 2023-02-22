import torch
from transformers import BartConfig, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import click
import numpy as np
import wandb
from data import prepare_dataset, load_data_file, create_encoder, write_predictions
from custom_tokenizers import word_tokenize
from encoder import MultiVocabularyEncoder
from eval import eval_morpheme_glosses
from datasets import DatasetDict

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_model(encoder: MultiVocabularyEncoder, sequence_length):
    print("Creating model...")
    config = BartConfig(
        vocab_size=encoder.vocab_size(),
        # d_model=100,
        # encoder_layers=3,
        # decoder_layers=3,
        # encoder_attention_heads=5,
        # decoder_attention_heads=5,
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        bos_token_id=encoder.BOS_ID,
        eos_token_id=encoder.EOS_ID,
        decoder_start_token_id=encoder.BOS_ID,
        forced_eos_token_id=encoder.EOS_ID,
        num_beams=5
    )
    model = BartForConditionalGeneration(config)
    print(model.config)
    return model.to(device)


def create_trainer(model: BartForConditionalGeneration, dataset, encoder: MultiVocabularyEncoder, batch_size, lr, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        decoded_preds = encoder.batch_decode(preds)

        # Decode (gold) labels
        labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels)
        return eval_morpheme_glosses(pred_morphemes=decoded_preds, gold_morphemes=decoded_labels)

    args = Seq2SeqTrainingArguments(
        f"../training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        predict_with_generate=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["dev"] if dataset else None,
        compute_metrics=compute_metrics
    )
    return trainer


languages = {
    'arp': 'Arapaho',
    'git': 'Gitksan',
    'lez': 'Lezgi',
    'nyb': 'Nyangbo',
    'ddo': 'Tsez',
    'usp': 'Uspanteko'
}

tokenizers = {
    'word': word_tokenize
}


@click.command()
@click.argument('mode')
@click.option("--tokenizer", help="word, bpe, or char", type=str, required=True)
@click.option("--lang", help="Which language to train", type=str, required=True)
@click.option("--pretrained_path", help="Path to pretrained model", type=click.Path(exists=True))
@click.option("--data_path", help="The dataset to run predictions on. Only valid in predict mode.", type=click.Path(exists=True))
def main(mode: str, tokenizer: str, lang: str, pretrained_path: str, data_path: str):
    if mode == 'train':
        wandb.init(project="igt-generation", entity="michael-ginn")

    MODEL_INPUT_LENGTH = 512

    train_data = load_data_file(f"../../GlossingSTPrivate/splits/{languages[lang]}/{lang}-train-track1-uncovered")
    dev_data = load_data_file(f"../../GlossingSTPrivate/splits/{languages[lang]}/{lang}-dev-track1-uncovered")

    print("Preparing datasets...")
    encoder = create_encoder(train_data, tokenizer=tokenizers[tokenizer], threshold=2)

    if mode == 'train':
        dataset = DatasetDict()
        dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizers[tokenizer], encoder=encoder,
                                           model_input_length=MODEL_INPUT_LENGTH, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizers[tokenizer], encoder=encoder,
                                         model_input_length=MODEL_INPUT_LENGTH, device=device)
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(model, dataset=dataset, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=100)

        print("Training...")
        trainer.train()
        print("Saving model to ./output")
        trainer.save_model('./output')
        print("Model saved at ./output")
    elif mode == 'predict':
        predict_data = load_data_file(data_path)
        predict_data = prepare_dataset(data=predict_data, tokenizer=tokenizers[tokenizer], encoder=encoder,
                                       model_input_length=MODEL_INPUT_LENGTH, device=device)
        model = BartForConditionalGeneration.from_pretrained(pretrained_path)
        trainer = create_trainer(model, dataset=None, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=100)
        preds = trainer.predict(test_dataset=predict_data)
        write_predictions(data_path, preds)


if __name__ == "__main__":
    main()
