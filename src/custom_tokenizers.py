import re

word_regex = r"[^.,!?;\s]+|[.,!?;]"
def word_tokenize(str: str):
    """Tokenizes by splitting into spaces, leaving punctuation as tokens"""
    return re.findall(word_regex, str)


word_regex_no_punc = r"[^.,!?;\s]+"
def word_tokenize_no_punc(str: str):
    """Tokenizes by splitting into spaces, skipping punctuation"""
    return re.findall(word_regex_no_punc, str)


tokenizers = {
    'word': word_tokenize,
    'word_no_punc': word_tokenize_no_punc
}