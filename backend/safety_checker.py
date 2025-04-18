from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
import time
from functools import lru_cache


class SafetyChecker:
    def __init__(self):
        # Initialize local model
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        self.model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        self.cache = {}
        self.cache_ttl = 3600  # Cache results for 1 hour

    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for k in expired_keys:
            del self.cache[k]

    @lru_cache(maxsize=1000)
    def __call__(self, text: str) -> Tuple[str, str]:
        """Check if content is safe and return detailed categorization
        
        Args:
            text (str): The text to check for inappropriate content
            
        Returns:
            Tuple[str, str]: A tuple containing:
                - The safety result ("safe" or "unsafe")
                - A detailed message about the classification
        """
        self._clean_cache()

        # Check cache first
        if text in self.cache:
            return self.cache[text][0], "cached"

        try:
            # Use local model for classification
            result = self.classifier(text)[0]
            is_safe = result['label'] == 'nothate'
            confidence = result['score']
            
            # Cache the result
            self.cache[text] = ("safe" if is_safe else "unsafe", time.time())
            
            if is_safe:
                return "safe", f"Content is appropriate (confidence: {confidence:.2f})"
            else:
                return "unsafe", f"Content may be inappropriate (confidence: {confidence:.2f})"
                
        except Exception as e:
            print(f"Error in local model classification: {e}")
            return "safe", "error"

    def check_transcription(self, text: str) -> Tuple[bool, str]:
        """Check transcribed text for inappropriate content
        
        Args:
            text (str): The transcribed text to check
            
        Returns:
            Tuple[bool, str]: A tuple containing:
                - Boolean indicating if the content is safe
                - A message explaining the result
        """
        result, details = self(text)
        if result == "unsafe":
            return False, f"Content moderation check failed: {details}"
        return True, "Content is appropriate"
