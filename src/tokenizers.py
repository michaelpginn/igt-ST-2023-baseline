import re

word_regex = r"[^.,!?;\s]+|[.,!?;]"
def word_tokenize(str: str):
    """Tokenizes by splitting into spaces"""
    return re.findall(word_regex, str)