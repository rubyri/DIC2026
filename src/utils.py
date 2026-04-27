import re

def load_stopwords(path="data/stopwords.txt"):
    """
    Reads the stopword list from a text file.

    Args:
        path (str): Relative path to the stopwords.txt file.

    Returns:
        set: A set of lowercase strings representing words to filter out.
    ```python
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def preprocess_text(text, stopwords_set):
    """
    Performs case folding, tokenization, and stopword filtering.
    Filters out tokens consisting of only one character.

    Args:
        text (str): The raw review text from the JSON input.
        stopwords_set (set): The set of words to be excluded.

    Returns:
        list: A list of tokens longer than 1 character that are not stopwords.
    """    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Strict Tokenization Delimiters
    # Includes: whitespace, digits, ()[]{}.!?,;:+=-_"'~#@&*%€$§\/<>`
    pattern = r"[\s\d\t\(\)\[\]\{\}\.\!\?\,\;\:\+\=\_\&\'\"\~\#\@\*\%\€\$\\\/\<\>\^\`\-]+"
    
    tokens = re.split(pattern, text)
    
    # 3. Filter: empty strings, stopwords, and tokens <= 1 character
    return [
        t for t in tokens 
        if t and t not in stopwords_set and len(t) > 1
    ]