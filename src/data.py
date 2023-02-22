"""Defines the data model.py and a function to easily load data"""
from typing import Optional, List
import re
from datasets import Dataset, DatasetDict
import torch
from custom_tokenizers import word_tokenize
from encoder import create_vocab, MultiVocabularyEncoder

class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: Optional[str], translation: str):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return None
        if not segmented:
            return self.glosses.split()
        else:
            return re.split("\s|-", self.glosses)

    def __dict__(self):
        d = {'transcription': self.transcription, 'translation': self.translation}
        if self.glosses is not None:
            d['glosses'] = self.gloss_list(segmented=True)
        return d


def load_data_file(path: str) -> List[IGTLine]:
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
            elif line_prefix == '\\g' and current_entry[2] == None and len(line[3:].strip()) > 0:
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


def create_encoder(train_data: List[IGTLine], threshold: int, tokenizer):
    """Creates an encoder with the vocabulary contained in train_data"""
    # Create the vocab for the source language
    source_data = [tokenizer(line.transcription) for line in train_data]
    source_vocab = create_vocab(source_data, threshold=threshold)

    # Create the shared vocab for the translation and glosses
    translation_data = [tokenizer(line.translation) for line in train_data]
    gloss_data = [line.gloss_list(segmented=True) for line in train_data]
    target_vocab = create_vocab(translation_data + gloss_data, threshold=threshold)

    # Create an encoder for both vocabularies
    return MultiVocabularyEncoder(vocabularies=[source_vocab, target_vocab])


def prepare_dataset(data: List[IGTLine], tokenizer, encoder: MultiVocabularyEncoder, model_input_length: int,  device):
    """Loads data, creates tokenizer, and creates a dataset object for easy manipulation"""

    # Create a dataset
    raw_dataset = Dataset.from_list([line.__dict__() for line in data])

    def process(row):
        source_enc = encoder.encode(tokenizer(row['transcription']), vocabulary_index=0)
        translation_enc = encoder.encode(tokenizer(row['translation']), vocabulary_index=1)
        combined_enc = source_enc + translation_enc

        # Pad
        initial_length = len(combined_enc)
        combined_enc += [encoder.PAD_ID] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (model_input_length - initial_length)

        # Encode the output, if present
        if 'glosses' in row:
            output_enc = encoder.encode(row['glosses'], vocabulary_index=1)
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
        else:
            return {'input_ids': torch.tensor(combined_enc).to(device),
                    'attention_mask': torch.tensor(attention_mask).to(device)}

    return raw_dataset.map(process)


def write_predictions(path: str, preds, encoder: MultiVocabularyEncoder):
    """Writes the predictions to a new file, which uses the file in `path` as input"""
    decoded_preds = encoder.batch_decode(preds)
    next_line = 0
    with open(path, 'r') as input:
        with open('output_preds', 'w') as output:
            for line in input:
                line_prefix = line[:2]
                if line_prefix == '\\g':
                    output_line = line_prefix + ' ' + ' '.join(decoded_preds[next_line])
                    output.write(output_line)
                    next_line += 1
                else:
                    output.write(line)
