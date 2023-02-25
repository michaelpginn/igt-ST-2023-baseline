import torch
from transformers import RobertaForTokenClassification, RobertaConfig, TrainingArguments, Trainer
import click
import numpy as np
import wandb
from data import prepare_dataset, load_data_file, create_encoder, write_predictions, ModelType
from custom_tokenizers import tokenizers
from encoder import MultiVocabularyEncoder, special_chars, load_encoder
from eval import eval_morpheme_glosses, eval_word_glosses
from datasets import DatasetDict
from typing import Optional

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_model(encoder: MultiVocabularyEncoder, sequence_length):
    print("Creating model...")
    config = RobertaConfig(
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[2]) + len(special_chars)
    )
    model = RobertaForTokenClassification(config)
    print(model.config)
    return model.to(device)


def create_trainer(model: RobertaForTokenClassification, dataset: Optional[DatasetDict], encoder: MultiVocabularyEncoder, batch_size, lr, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.argmax(preds, axis=2)

        # Decode predicted output
        decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=2)
        print(decoded_preds[0:1])

        # Decode (gold) labels
        labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=2)
        print(decoded_labels[0:1])
        return eval_word_glosses(pred_words=decoded_preds, gold_words=decoded_labels)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        eval_accumulation_steps=6,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
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


@click.command()
@click.argument('mode')
@click.option("--lang", help="Which language to train", type=str, required=True)
@click.option("--pretrained_path", help="Path to pretrained model", type=click.Path(exists=True))
@click.option("--encoder_path", help="Path to pretrained encoder", type=click.Path(exists=True))
@click.option("--data_path", help="The dataset to run predictions on. Only valid in predict mode.", type=click.Path(exists=True))
def main(mode: str, lang: str, pretrained_path: str, encoder_path: str, data_path: str):
    if mode == 'train':
        wandb.init(project="igt-generation", entity="michael-ginn")

    MODEL_INPUT_LENGTH = 512

    train_data = load_data_file(f"../../GlossingSTPrivate/splits/{languages[lang]}/{lang}-train-track1-uncovered")
    dev_data = load_data_file(f"../../GlossingSTPrivate/splits/{languages[lang]}/{lang}-dev-track1-uncovered")

    print("Preparing datasets...")

    if mode == 'train':
        encoder = create_encoder(train_data, tokenizer=tokenizers['word_no_punc'], threshold=1,
                                 model_type=ModelType.TOKEN_CLASS)
        encoder.save()
        dataset = DatasetDict()
        dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizers['word_no_punc'], encoder=encoder,
                                           model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizers['word_no_punc'], encoder=encoder,
                                         model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(model, dataset=dataset, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=50)

        print("Training...")
        trainer.train()
        print("Saving model to ./output")
        trainer.save_model('./output')
        print("Model saved at ./output")
    elif mode == 'predict':
        encoder = load_encoder(encoder_path)
        predict_data = load_data_file(data_path)
        predict_data = prepare_dataset(data=predict_data, tokenizer=tokenizers['word_no_punc'], encoder=encoder,
                                       model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        model = RobertaForTokenClassification.from_pretrained(pretrained_path)
        trainer = create_trainer(model, dataset=None, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=50)
        preds = trainer.predict(test_dataset=predict_data).predictions
        preds = np.argmax(preds, axis=2)
        write_predictions(data_path, preds, encoder=encoder, from_vocabulary_index=2)


if __name__ == "__main__":
    main()
