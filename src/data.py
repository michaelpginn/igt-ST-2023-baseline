"""Defines the data model.py and a function to easily load data"""
from typing import Optional
import re
from datasets import Dataset, DatasetDict
import torch
from tokenizers import word_tokenize
from encoder import create_vocab, MultiVocabularyEncoder

class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: str, translation: str):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False):
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if not segmented:
            return self.glosses.split()
        else:
            return re.split("\s|-", self.glosses)


def load_data_file(path: str) -> list[IGTLine]:
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\g' and current_entry[2] == None:
                current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line.strip() != "":
                # Something went wrong
                print("Skipping line: ", line)
                current_entry = [None, None, None, None]
    return all_data


def prepare_dataset(train_path: str, dev_path: str, tokenizer, model_input_length: int, device):
    """Loads data, creates tokenizer, and creates a dataset object for easy manipulation"""
    train_data = load_data_file(train_path)

    # Create the vocab for the source language
    source_data = [tokenizer(line.transcription) for line in train_data]
    source_vocab = create_vocab(source_data)

    # Create the shared vocab for the translation and glosses
    translation_data = [tokenizer(line.translation) for line in train_data]
    gloss_data = [line.gloss_list(segmented=True) for line in train_data]
    target_vocab = create_vocab(translation_data + gloss_data)

    dev_data = load_data_file(dev_path)

    # Create a dataset
    raw_dataset = DatasetDict()
    raw_dataset['train'] = Dataset.from_list([{'igt': line} for line in train_data])
    raw_dataset['dev'] = Dataset.from_list([{'igt': line} for line in dev_data])

    # Create an encoder for both vocabularies
    encoder = MultiVocabularyEncoder(vocabularies=[source_vocab, target_vocab])

    def process(row):
        source_enc = encoder.encode(tokenizer(row['igt'].transcription), vocabulary_index=0)
        translation_enc = encoder.encode(tokenizer(row['igt'].translation), vocabulary_index=1)
        combined_enc = source_enc + translation_enc

        # Pad
        initial_length = len(combined_enc)
        combined_enc += [encoder.PAD_ID] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (model_input_length - initial_length)

        # Encode the output
        output_enc = encoder.encode(row['igt'].gloss_list(segmented=True), vocabulary_index=1)
        output_enc = output_enc + [encoder.EOS_ID]

        # Shift one position right
        decoder_input_ids = [encoder.BOS_ID] + output_enc

        # Pad both
        output_enc += [encoder.PAD_ID] * (model_input_length - len(output_enc))
        decoder_input_ids += [encoder.PAD_ID] * (model_input_length - len(decoder_input_ids))

        return {'input_ids': torch.tensor(combined_enc).to(device),
                'attention_mask': torch.tensor(attention_mask).to(device),
                'labels': torch.tensor(output_enc).to(device),
                'decoder_input_ids': torch.tensor(decoder_input_ids).to(device)}

    dataset = DatasetDict()
    dataset['train'] = raw_dataset['train'].map(process)
    dataset['dev'] = raw_dataset['dev'].map(process)
    return dataset, encoder
