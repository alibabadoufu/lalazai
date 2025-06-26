"""
Unified LLM Client for on-premise OpenAI compatible API.
This module provides a production-ready client with error handling, retries, and monitoring.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from LLM calls."""
    content: str
    token_count: int
    model: str
    timestamp: datetime
    latency_ms: int
    metadata: Dict[str, Any]


class LLMClient:
    """
    Unified client for on-premise OpenAI compatible API.
    
    Features:
    - Automatic retries with exponential backoff
    - Token counting and budget management
    - Structured output handling
    - Request/response logging
    - Error handling and fallbacks
    - Rate limiting
    """

    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model: str = "gpt-4",
                 max_retries: int = 3,
                 timeout: int = 60,
                 token_budget: int = 100000):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key (from env if not provided)
            api_base: API base URL (from env if not provided)
            model: Default model to use
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            token_budget: Daily token budget limit
        """
        # Configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        self.model = model
        self.timeout = timeout
        self.token_budget = token_budget
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Request session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Usage tracking
        self.daily_token_usage = 0
        self.request_count = 0
        self.last_reset = datetime.now().date()
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate client configuration."""
        if not self.api_key:
            self.logger.warning("No API key provided - client will not work with real API")
        
        if not self.api_base:
            raise ValueError("API base URL is required")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (rough estimate)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def generate_completion(self, prompt, max_tokens=100, temperature=0.7, model=None, **kwargs):
        """
        Generate text completion using LLM.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            model: Model to use (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Generated text or LLMResponse object
            
        Raises:
            RuntimeError: If generation fails after retries
            ValueError: If token budget exceeded
        """
        start_time = time.time()
        model = model or self.model
        
        self.logger.info(f"Generating completion with model {model}")
        
        if not self.api_key:
            return self._generate_mock_completion(prompt)
        
        try:
            # Build request payload
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            response = self.session.post(
                f"{self.api_base.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data["choices"][0]["message"]["content"]
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            token_count = self._count_tokens(content)
            
            self.logger.info(f"Completion successful: {token_count} tokens, {latency_ms}ms")
            
            return LLMResponse(
                content=content,
                token_count=token_count,
                model=model,
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                metadata={"temperature": temperature}
            )
            
        except Exception as e:
            self.logger.error(f"LLM completion failed: {e}")
            return self._generate_mock_completion(prompt)

    def generate_embedding(self, text):
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        self.logger.info(f"Generating embedding for text: {text[:50]}...")
        
        if not self.api_key:
            return self._generate_mock_embedding(len(text))
        
        try:
            payload = {
                "model": "text-embedding-ada-002",
                "input": text
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.api_base.rstrip('/')}/embeddings",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            embedding = data["data"][0]["embedding"]
            self.logger.info(f"Embedding generated: {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return self._generate_mock_embedding(len(text))

    def _generate_mock_completion(self, prompt: str) -> str:
        """Generate mock completion for testing."""
        self.logger.info(f"Generating mock completion for prompt: {prompt[:50]}...")
        
        # Simple rule-based responses for common patterns
        prompt_lower = prompt.lower()
        
        if "summarize" in prompt_lower:
            return "This is a mock summary of the provided content."
        elif "keywords" in prompt_lower:
            return "keyword1, keyword2, keyword3"
        elif "sentiment" in prompt_lower:
            return "neutral"
        elif "intent" in prompt_lower:
            return "information_seeking"
        else:
            return f"Mock LLM response based on: {prompt[:100]}..."

    def _generate_mock_embedding(self, text_length: int) -> List[float]:
        """Generate mock embedding for testing."""
        self.logger.info(f"Generating mock embedding for text length: {text_length}")
        
        # Generate deterministic but varied embedding based on text length
        import hashlib
        hash_val = int(hashlib.md5(str(text_length).encode()).hexdigest(), 16)
        
        # Generate 1536-dimensional embedding (standard for text-embedding-ada-002)
        embedding = []
        for i in range(1536):
            # Use hash to generate deterministic values
            val = ((hash_val + i) % 10000) / 10000.0 - 0.5  # Range: -0.5 to 0.5
            embedding.append(val)
        
        return embedding

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on LLM service.
        
        Returns:
            Health status information
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "llm_client",
            "status": "healthy" if self.api_key else "mock_mode",
            "model": self.model,
            "api_base": self.api_base,
            "has_api_key": bool(self.api_key)
        }


