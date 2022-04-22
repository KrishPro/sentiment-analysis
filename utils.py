"""
Written by KrishPro @ KP
"""

import re

def process_text(text: str):
    """
    This removes
     - URLs
     - Usernames
     - Punctations (replace them with  ' ')
     - Numbers (replace them with ' ')
     - Multiple Spaces together (replace them with ' ')
    """
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r' www\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = text.strip().lower().encode('ascii', 'ignore').decode()
    return text