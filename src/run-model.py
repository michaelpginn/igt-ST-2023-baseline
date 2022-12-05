import word_model
import char_model
import bpe_model
from transformers import BartForConditionalGeneration
import torch
import sys


def preprocess(encoder, words, translation):
    """Preprocesses input for prediction
    """

    SEP_ID = encoder.all_vocab.all_vocabecial_chars.index("[SEP]")

    source_enc = encoder.encode(words)
    transl_enc = encoder.encode(translation, vocab='transl')
    combined_enc = source_enc + [SEP_ID] + transl_enc

    return torch.tensor(combined_enc)


def preprocess_bpe(tokenizer, words, translation):
    source_enc = tokenizer.encode(words, is_split_into_words=False, add_special_tokens=False)
    transl_enc = tokenizer.encode(translation, is_split_into_words=False, add_special_tokens=False)
    combined_enc = source_enc + [tokenizer.sep_token_id] + transl_enc

    return torch.tensor(combined_enc)


def main():

    path = './output'
    model_input_length = 512

    if len(sys.argv) == 2:
        # Eval mode
        is_eval = True
    else:
        words = sys.argv[2]
        translation = sys.argv[3]

    model_type = sys.argv[1]

    if model_type == 'word':
        dataset, vocab_size, encoder = word_model.prepare_data(paths=['../data/kor.xml'], model_input_length=model_input_length)
        model = BartForConditionalGeneration.from_pretrained(path)
        trainer = word_model.create_trainer(model, dataset, encoder, batch_size=16, lr=2e-5, max_epochs=20)

        if is_eval:
            print(trainer.evaluate())
        else:
            print(model.generate(preprocess(encoder, words.split(), translation.split())))
    elif model_type == 'char':
        dataset, vocab_size, encoder = char_model.prepare_data(paths=['../data/kor.xml'], model_input_length=model_input_length)
        model = BartForConditionalGeneration.from_pretrained(path)
        trainer = char_model.create_trainer(model, dataset, encoder, batch_size=16, lr=2e-5, max_epochs=20)
        if is_eval:
            print(trainer.evaluate())
        else:
            print(model.generate(preprocess(encoder, words.split(), translation.split())))
    elif model_type == 'bpe':
        dataset, vocab_size, tokenizer = bpe_model.prepare_data(paths=['../data/kor.xml'], model_input_length=model_input_length)
        model = BartForConditionalGeneration.from_pretrained(path)
        trainer = bpe_model.create_trainer(model, dataset, tokenizer, batch_size=16, lr=2e-5, max_epochs=20)
        if is_eval:
            print(trainer.evaluate())
        else:
            print(model.generate(preprocess(tokenizer, words, translation)))


if __name__ == "__main__":
    main()
