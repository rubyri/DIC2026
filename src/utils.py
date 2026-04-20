import re

def load_stopwords(path="data/stopwords.txt"):
    """Reads the TUWEL stopword list and returns a set."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def preprocess_text(text, stopwords_set):
    # 1. Case folding
    text = text.lower()
    
    # 2. Strict Tokenization Delimiters
    # Includes: whitespace, digits, ()[]{}.!?,;:+=-_"'~#@&*%€$§\/<>`
    pattern = r"[\s\d\(\)\[\]\{\}\.\!\?\,\;\:\+\=\-\_\&\'\"\~\#\@\*\%\€\$\§\/\<\>\\\`\^]+"
    
    tokens = re.split(pattern, text)
    
    # 3. Filter: empty strings, stopwords, and tokens <= 1 character
    return [
        t for t in tokens 
        if t and t not in stopwords_set and len(t) > 1
    ]