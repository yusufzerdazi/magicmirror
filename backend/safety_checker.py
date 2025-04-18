from openai import OpenAI
import os
from dotenv import load_dotenv
import concurrent.futures
from typing import Dict, Optional, Tuple
import time
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class SafetyChecker:
    def __init__(self, use_local_model: bool = True):
        load_dotenv()
        self.use_local_model = use_local_model
        
        if use_local_model:
            # Initialize local model
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
            self.model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            
        self.cache: Dict[str, Tuple[str, float]] = {}
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
    def __call__(self, prompt: str) -> Tuple[str, str]:
        """Check if content is safe and return detailed categorization"""
        self._clean_cache()

        # Check cache first
        if prompt in self.cache:
            return self.cache[prompt][0], "cached"

        if self.use_local_model:
            try:
                # Use local model for classification
                result = self.classifier(prompt)[0]
                is_safe = result['label'] == 'nothate'
                confidence = result['score']
                
                # Cache the result
                self.cache[prompt] = ("safe" if is_safe else "unsafe", time.time())
                
                if is_safe:
                    return "safe", f"Content is appropriate (confidence: {confidence:.2f})"
                else:
                    return "unsafe", f"Content may be inappropriate (confidence: {confidence:.2f})"
                    
            except Exception as e:
                print(f"Error in local model classification: {e}")
                return "safe", "error"
        else:
            # Use GPT-4 for classification
            def make_api_call(content: str) -> str:
                response = self.client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a content moderation AI that analyzes text for inappropriate content. 
                            For each input, respond with a JSON object containing:
                            {
                                "safe": boolean,
                                "categories": string[],
                                "reason": string,
                                "confidence": float
                            }
                            Categories can include: violence, nudity, sexual, hate, illegal, harmful, other
                            Confidence should be between 0 and 1."""
                        },
                        {"role": "user", "content": content}
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

            content = f'Analyze this text for inappropriate content: "{prompt}"'

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(make_api_call, content)
                try:
                    result = future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    return "safe", "timeout"

            try:
                import json
                analysis = json.loads(result)
                
                # Cache the result
                self.cache[prompt] = (analysis["safe"], time.time())
                
                if analysis["safe"]:
                    return "safe", "analyzed"
                else:
                    return "unsafe", f"Detected categories: {', '.join(analysis['categories'])}. Reason: {analysis['reason']}"
                    
            except Exception as e:
                print(f"Error parsing safety check result: {e}")
                return "safe", "error"

    def check_transcription(self, text: str) -> Tuple[bool, str]:
        """Check transcribed text for inappropriate content"""
        result, details = self(text)
        if result == "unsafe":
            return False, f"Content moderation check failed: {details}"
        return True, "Content is appropriate"
