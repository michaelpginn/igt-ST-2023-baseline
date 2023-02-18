"""Defines a tokenizer that uses two distinct vocabulary"""
from typing import List
import torch

special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]


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


class MultiVocabularyEncoder():
    """Encodes and decodes words to an integer representation"""

    def __init__(self, vocabularies: List[List[str]]):
        """
        :param vocabularies: A list of vocabularies for the tokenizer
        """
        self.vocabularies = vocabularies
        self.all_vocab = special_chars + sum(self.vocabularies, [])

        self.PAD_ID = special_chars.index("[PAD]")
        self.SEP_ID = special_chars.index("[SEP]")
        self.BOS_ID = special_chars.index("[BOS]")
        self.EOS_ID = special_chars.index("[EOS]")

    def encode_word(self, word, vocabulary_index):
        """Converts a word to the integer encoding"""
        if not word.isupper():
            word = word.lower()

        if word in special_chars:
            return special_chars.index(word)
        elif vocabulary_index < len(self.vocabularies):
            prior_vocab_padding = len(sum(self.vocabularies[:vocabulary_index], [])) # Sums the length of all preceding vocabularies
            if word in self.vocabularies[vocabulary_index]:
                return self.vocabularies[vocabulary_index].index(word) + prior_vocab_padding + len(special_chars)
            else:
                return 0
        else:
            # We got a bad vocabulary
            raise ValueError('VOCABULARY BAD! BOO! HISS!! GAAHH!')

    def encode(self, sentence: List[str], vocabulary_index) -> List[int]:
        """Encodes a sentence (a list of strings)"""
        return [self.encode_word(word, vocabulary_index=vocabulary_index) for word in sentence]

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