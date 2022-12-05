from tokenizers.implementations import ByteLevelBPETokenizer
from xigt.codecs import xigtxml
from typing import List
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset, DatasetDict
from transformers import BartConfig, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartTokenizer, DataCollatorForSeq2Seq
from torchtext.data.metrics import bleu_score
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MissingValueError(Exception):
    pass

def extract_igt(igt):
    """From a single line of IGT, extracts the features which are allowed in this shared task:
    1. Transcribed words (not segmented)
    2. Translation (not aligned)
    3. Glosses"""
    if not igt.get('w'):
        raise MissingValueError("words")
    if not igt.get('tw'):
        raise MissingValueError("translation")
    if not igt.get('gw'):
        raise MissingValueError("glosses")
        
    words = [word.value() for word in igt['w'].items]
    glosses = [gloss.value() for gloss in igt['gw'].items]
    
    translation = [item.value() for item in igt['tw']]
    return {'words': words, 'translation': translation, 'glosses': glosses}
    
    
def load_preprocess_data(path):
    """Loads and preprocesses the data from an IGTXML file"""
    
    corpus = xigtxml.load(open(path))
    corpus_data = []

    missing_words_count = 0
    missing_translation_count = 0
    missing_gloss_count = 0
    all_good_count = 0

    # Extract igt for each item in the corpus, removing those that are missing info
    for i, igt in enumerate(corpus):
        try:
            igt_data = extract_igt(igt)
            corpus_data.append(igt_data)
            all_good_count += 1
        except MissingValueError as v:
            if str(v) == 'words': 
                missing_words_count += 1
            elif str(v) == 'translation':
               missing_translation_count += 1
            elif str(v) == 'glosses':
                missing_gloss_count += 1

    print(f"Parsed corpus, with \n\t{all_good_count} good rows\n\t{missing_words_count} rows missing words\
            \n\t{missing_translation_count} missing translations\n\t{missing_gloss_count} missing glosses")
    
    # Remove the dashes from the input, to simulate the case where we don't have segmentation
    for item in corpus_data:
        for i, word in enumerate(item['words']):
            item['words'][i] = word.replace('-', '')
            
    # Split the output by dashes, so each gloss is a single item
    for item in corpus_data:
        glosses = []
        for i, word in enumerate(item['glosses']):
            word_glosses = word.split("-")
            glosses.append(word_glosses[0])
            glosses += ["-" + gloss for gloss in word_glosses[1:]]
        item['glosses'] = glosses
        
    return corpus_data


special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]

def convert_to_dataset(tokenizer, train, dev, test, model_input_length):
    """Converts raw lists of data into a Dataset of encoded indices"""

    raw_dataset = DatasetDict()
    raw_dataset['train'] = Dataset.from_list(train)
    raw_dataset['validation'] = Dataset.from_list(dev)
    raw_dataset['test'] = Dataset.from_list(test)
    
    def preprocess(row):
        """Preprocesses each row in the dataset
        1. Combines the source and translation into a single list, and encodes
        2. Pads the combined input and output sequences
        3. Creates attention mask
        """
        source_enc = tokenizer.encode(row['words'], is_split_into_words=True, add_special_tokens=False)
        transl_enc = tokenizer.encode(row['translation'], is_split_into_words=True, add_special_tokens=False)
        combined_enc = source_enc + [tokenizer.sep_token_id] + transl_enc

        attention_mask = [1] * len(combined_enc)

        # Encode the output
        output_enc = tokenizer.encode(row['glosses'], is_split_into_words=True, add_special_tokens=False)
        output_enc = [tokenizer.bos_token_id] + output_enc + [tokenizer.eos_token_id]

        return {'input_ids': torch.tensor(combined_enc).to(device), 
                'attention_mask': torch.tensor(attention_mask).to(device), 
                'labels': torch.tensor(output_enc).to(device)}
    
    dataset = DatasetDict()
    dataset['train'] = raw_dataset['train'].map(preprocess)
    dataset['validation'] = raw_dataset['validation'].map(preprocess)
    dataset['test'] = raw_dataset['test'].map(preprocess)
    
    return dataset


def prepare_data(paths: List[str], model_input_length: int):
    """Master function to load and prepare the data as a Dataset"""
    print("Loading data...")
    corpus_data = []
    for path in paths:
        corpus_data += load_preprocess_data(path)
    
    print("Creating tokenizer...")

    all_text = open('all_text.txt', 'w')
    all_text.write('')
    all_text.close()
    all_text = open('all_text.txt', 'a')
    for item in corpus_data:
        all_text.write(" ".join(item['words']))
        all_text.write(" ".join(item['translation']))
        all_text.write(" ".join(item['glosses']))

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=['./all_text.txt'], min_frequency=2, special_tokens=special_chars)
    tokenizer.save_model(".", "tokenizer")
    tokenizer = BartTokenizer('./tokenizer-vocab.json', './tokenizer-merges.txt', bos_token="[BOS]", eos_token="[EOS]",
                              sep_token="[SEP]", cls_token="[BOS]", unk_token="[UNK]", pad_token="[PAD]",
                              mask_token="[MASK]", model_max_length=model_input_length)
    
    train, test = train_test_split(corpus_data, test_size=0.3)
    test, dev = train_test_split(test, test_size=0.5)

    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")
    
    print("Creating dataset...")
    dataset = convert_to_dataset(tokenizer, train, dev, test, model_input_length)
    vocab_size = tokenizer.vocab_size
    
    return dataset, vocab_size, tokenizer
        
    
def create_model(vocab_size, tokenizer: BartTokenizer, sequence_length=512):
    print("Creating model...")
    config = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        num_beams=5
    )
    model = BartForConditionalGeneration(config)
    print(model.config)
    return model.to(device)
    
    
def create_trainer(model, dataset, tokenizer: BartTokenizer, batch_size=16, lr=2e-5, max_epochs=20):
    print("Creating trainer...")

    def postprocess_text(preds, labels):
        preds = [pred.strip().split() for pred in preds]
        labels = [[label.strip().split()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode (gold) labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        bleu = bleu_score(decoded_preds, [[seq] for seq in decoded_labels])

        # Also get accuracy, based on (correct morphemes in output) / (len of correct output)
        correct_glosses = 0
        total_glosses = 0

        for (pred, labels) in zip(decoded_preds, decoded_labels):
            correct_glosses += len([gloss for gloss in labels if gloss in pred ])
            total_glosses += len(labels)

        acc = round(correct_glosses / total_glosses, 4)

        return {'bleu': bleu, 'accuracy': acc}
    
    args = Seq2SeqTrainingArguments(
        f"training-checkpoints",
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
        # fp16=True,
    )
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics
    )
    return trainer
    
    
# Actual script
def main():
    model_input_length = 512
    dataset, vocab_size, tokenizer = prepare_data(paths=['../data/kor.xml'], model_input_length=model_input_length)
    model = create_model(vocab_size=vocab_size, tokenizer=tokenizer, sequence_length=model_input_length)
    trainer = create_trainer(model, dataset, tokenizer, batch_size=16, lr=2e-5, max_epochs=20)
    print("Training...")
    trainer.train()
    print("Saving model to ./output")
    trainer.save_model('./output')
    print("Model saved at ./output")

if __name__ == "__main__":
    main()
