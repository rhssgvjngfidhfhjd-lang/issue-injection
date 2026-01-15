import re
import difflib

# 计算文本中的单词数量，处理各种分隔符和标点符号。
def count_words(text: str) -> int:
    """Count words in text, handling various separators and punctuation."""
    if not text or not text.strip():
        return 0
    
    # Remove extra whitespace and split by common word separators
    words = re.findall(r'\b\w+\b', text)
    return len(words)


# define the origin text word limit
# 将文本截断到指定的单词限制，同时保留句子边界。
def truncate_to_word_limit(text: str, max_words: int = 500) -> str:
    """Truncate text to specified word limit while preserving sentence boundaries."""
    if not text or not text.strip():
        return text
    
    words = re.findall(r'\b\w+\b', text)
    if len(words) <= max_words:
        return text
    
    # Find the position of the last word within the limit
    if max_words == 0:
        return ""
    
    # Find the position of the max_words-th word in the original text
    word_positions = []
    for match in re.finditer(r'\b\w+\b', text):
        word_positions.append((match.start(), match.end()))
    
    if len(word_positions) < max_words:
        return text
    
    # Get the end position of the max_words-th word
    end_pos = word_positions[max_words - 1][1]
    truncated_text = text[:end_pos]
    
    # Try to end at a sentence boundary
    last_period = truncated_text.rfind('.')
    last_exclamation = truncated_text.rfind('!')
    last_question = truncated_text.rfind('?')
    
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > len(truncated_text) * 0.7:  # If we can end at a sentence boundary
        return truncated_text[:last_sentence_end + 1]
    else:
        return truncated_text + "..."


# 使用 difflib SequenceMatcher 计算两个文本之间的相似度。
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using difflib SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts for comparison
    text1_normalized = re.sub(r'\s+', ' ', text1.strip().lower())
    text2_normalized = re.sub(r'\s+', ' ', text2.strip().lower())
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, text1_normalized, text2_normalized).ratio()
    return similarity


# 检查修改后的文本是否满足与原始文本的相似度阈值。
def check_similarity_threshold(original: str, modified: str, threshold: float = 0.6) -> bool:
    """Check if modified text meets similarity threshold with original text."""
    similarity = calculate_similarity(original, modified)
    return similarity >= threshold
