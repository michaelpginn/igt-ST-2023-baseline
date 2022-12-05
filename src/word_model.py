from xigt.codecs import xigtxml
from typing import List
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset, DatasetDict
from transformers import BartConfig, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
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

def create_vocab(sentences: List[List[str]], threshold=2):
    """Creates a set of the unique words in a list of sentences, only including words that exceed the threshold"""
    all_words = dict()
    for sentence in sentences:
        for word in sentence:
            all_words[word.lower()] = all_words.get(word.lower(), 0) + 1

    all_words_list = []
    for word, count in all_words.items():
        if count >= threshold:
            all_words_list.append(word)

    return sorted(all_words_list)


special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]

class IntegerEncoder():
    """Encodes and decodes words to an integer representation"""
    def __init__(self, source_vocab, target_and_gloss_vocab):
        self.source_vocab = source_vocab
        self.target_and_gloss_vocab = target_and_gloss_vocab
        self.all_vocab = special_chars + source_vocab + target_and_gloss_vocab

    
    def encode_word(self, word, vocab='source'):
        """Converts a word to the integer encoding"""
        word = word.lower()

        if word in special_chars:
            return special_chars.index(word)
        if vocab == 'source':
            if word in self.source_vocab:
                return self.source_vocab.index(word) + len(special_chars)
            else:
                return 0
        else:
            if word in self.target_and_gloss_vocab:
                return self.target_and_gloss_vocab.index(word) + len(special_chars) + len(self.source_vocab)
            else:
                return 0
            
    def encode(self, sentence: List[str], vocab='source') -> List[int]:
        """Encodes a sentence (a list of strings)"""
        return [self.encode_word(word, vocab=vocab) for word in sentence]
    

    def batch_decode(self, batch):
        """Decodes a batch of indices to the actual words"""
        def decode(seq):
            if isinstance(seq, torch.Tensor):
                indices = seq.detach().cpu().tolist()
            else:
                indices = seq.tolist()
            return [self.all_vocab[index] for index in indices if index >= len(special_chars)]

        return [decode(seq) for seq in batch]
    
    def vocab_size(self):
        return len(self.all_vocab)


PAD_ID = special_chars.index("[PAD]")
SEP_ID = special_chars.index("[SEP]")
BOS_ID = special_chars.index("[BOS]")
EOS_ID = special_chars.index("[EOS]")


def convert_to_dataset(encoder, train, dev, test, model_input_length):
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
        source_enc = encoder.encode(row['words'])
        transl_enc = encoder.encode(row['translation'], vocab='transl')
        combined_enc = source_enc + [SEP_ID] + transl_enc

        # Pad
        initial_length = len(combined_enc)
        combined_enc += [PAD_ID] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (model_input_length - initial_length)

        # Encode the output
        output_enc = encoder.encode(row['glosses'], vocab='transl')
        output_enc = output_enc + [EOS_ID]

        # Shift one position right
        decoder_input_ids = [BOS_ID] + output_enc

        # Pad both
        output_enc += [PAD_ID] * (model_input_length - len(output_enc))
        decoder_input_ids += [PAD_ID] * (model_input_length - len(decoder_input_ids))

        return {'input_ids': torch.tensor(combined_enc).to(device),
                'attention_mask': torch.tensor(attention_mask).to(device),
                'labels': torch.tensor(output_enc).to(device),
                'decoder_input_ids': torch.tensor(decoder_input_ids).to(device)}
    
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
    
    print("Creating vocabulary...")
    source_vocab = create_vocab([item['words'] for item in corpus_data])
    target_and_gloss_vocab = create_vocab([item['translation'] for item in corpus_data] + [item['glosses'] for item in corpus_data])
    
    encoder = IntegerEncoder(source_vocab, target_and_gloss_vocab)
    
    train, test = train_test_split(corpus_data, test_size=0.3)
    test, dev = train_test_split(test, test_size=0.5)

    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")
    
    print("Creating dataset...")
    dataset = convert_to_dataset(encoder, train, dev, test, model_input_length)
    vocab_size = encoder.vocab_size()
    
    return dataset, vocab_size, encoder
        
    
def create_model(vocab_size, sequence_length=512):
    print("Creating model...")
    config = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        decoder_start_token_id=BOS_ID,
        forced_eos_token_id=EOS_ID,
        num_beams = 5
    )
    model = BartForConditionalGeneration(config)
    print(model.config)
    return model.to(device)
    
    
def create_trainer(model, dataset, encoder: IntegerEncoder, batch_size=16, lr=2e-5, max_epochs=20):
    print("Creating trainer...")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        decoded_preds = encoder.batch_decode(preds)

        # Decode (gold) labels
        labels = np.where(labels != -100, labels, PAD_ID)
        decoded_labels = encoder.batch_decode(labels)

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
        compute_metrics=compute_metrics
    )
    return trainer
    
    
# Actual script
def main():
    model_input_length = 512
    dataset, vocab_size, encoder = prepare_data(paths=['../data/kor.xml'], model_input_length=model_input_length)
    model = create_model(vocab_size=vocab_size, sequence_length=model_input_length)
    trainer = create_trainer(model, dataset, encoder, batch_size=16, lr=2e-5, max_epochs=20)
    print("Training...")
    trainer.train()
    print("Saving model to ./output")
    trainer.save_model('./output')
    print("Model saved at ./output")

if __name__ == "__main__":
    main()
