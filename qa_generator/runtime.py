"""
Pluggable LLM runtime abstraction for QA generator.

Provides unified interface for local mlx-llm-server and hosted APIs (OpenAI, Anthropic).
Handles auto-detection, fallback, and deterministic JSON generation.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Protocol
from abc import ABC, abstractmethod
import json
import requests
from openai import OpenAI
import logging
from .exceptions import LLMRuntimeError, ConfigurationError

logger = logging.getLogger(__name__)

# Import the simple MLX runtime
try:
    from .simple_mlx_runtime import SimpleMLXRuntime
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class LLMRuntime(Protocol):
    """Protocol for all LLM runtime implementations."""
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text response from prompt."""
        ...
    
    def is_available(self) -> bool:
        """Check if this runtime is currently available."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        ...


class OpenAICompatibleRuntime:
    """
    Unified runtime for OpenAI-compatible APIs.
    Works with local mlx-llm-server, OpenAI, and other compatible services.
    """
    
    def __init__(
        self, 
        base_url: str,
        api_key: str = "no-key",
        model: str = "gpt-3.5-turbo",
        name: str = "openai-compatible"
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.name = name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate response using OpenAI-compatible API."""
        try:
            # Ensure deterministic JSON responses
            system_prompt = (
                "You are a QA test scenario generator. Return JSON only matching the provided schema. "
                "No markdown formatting, no explanations, just valid JSON."
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise LLMRuntimeError(f"Failed to generate response from {self.name}: {e}")
    
    def is_available(self) -> bool:
        """Check if the service is reachable."""
        try:
            # Try to reach the health endpoint or models endpoint
            health_url = f"{self.base_url.rstrip('/v1')}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            try:
                # Fallback: try models endpoint
                models_url = f"{self.base_url}/models"
                response = requests.get(
                    models_url, 
                    headers={"Authorization": f"Bearer {self.api_key}"}, 
                    timeout=5
                )
                return response.status_code == 200
            except requests.RequestException:
                return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "type": "openai-compatible"
        }


class RuntimeFactory:
    """Factory for creating and managing LLM runtimes with auto-detection."""
    
    DEFAULT_LOCAL_URL = "http://localhost:8080/v1"
    
    def __init__(self):
        self._runtimes = {}
        self._setup_default_runtimes()
    
    def _setup_default_runtimes(self):
        """Setup default runtime configurations."""
        # Local MLX - simple direct approach
        if HAS_MLX:
            self._runtimes["local"] = {
                "class": SimpleMLXRuntime,
                "kwargs": {
                    "model_path": "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"
                }
            }
        else:
            # Fallback to OpenAI-compatible if MLX not available
            self._runtimes["local"] = {
                "class": OpenAICompatibleRuntime,
                "kwargs": {
                    "base_url": self.DEFAULT_LOCAL_URL,
                    "api_key": "no-key",
                    "model": "local-model",
                    "name": "mlx-fallback"
                }
            }
        
        # OpenAI (requires API key)
        self._runtimes["openai"] = {
            "class": OpenAICompatibleRuntime,
            "kwargs": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "name": "openai"
            }
        }
        
        # Anthropic via OpenAI-compatible proxy (if available)
        self._runtimes["anthropic"] = {
            "class": OpenAICompatibleRuntime,
            "kwargs": {
                "base_url": "https://api.anthropic.com/v1",  # Hypothetical
                "model": "claude-3-sonnet",
                "name": "anthropic"
            }
        }
    
    def register_runtime(self, name: str, runtime_class, **kwargs):
        """Register a custom runtime."""
        self._runtimes[name] = {
            "class": runtime_class,
            "kwargs": kwargs
        }
    
    def create_runtime(
        self, 
        preferred: Optional[str] = None,
        local_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_fallback: bool = True
    ) -> LLMRuntime:
        """
        Create the best available runtime.
        
        Args:
            preferred: Preferred runtime name ("local", "openai", etc.)
            local_url: Custom local server URL
            api_key: API key for hosted services
            auto_fallback: Whether to fallback to available runtimes
        """
        
        # Update local URL if provided
        if local_url:
            self._runtimes["local"]["kwargs"]["base_url"] = local_url
        
        # Update API keys for hosted services
        if api_key:
            for runtime_name in ["openai", "anthropic"]:
                if runtime_name in self._runtimes:
                    self._runtimes[runtime_name]["kwargs"]["api_key"] = api_key
        
        # Try preferred runtime first
        if preferred and preferred in self._runtimes:
            runtime = self._create_runtime_instance(preferred)
            if runtime and runtime.is_available():
                logger.info(f"Using preferred runtime: {preferred}")
                return runtime
            elif not auto_fallback:
                raise LLMRuntimeError(f"Preferred runtime '{preferred}' is not available")
        
        # Auto-detect available runtime
        if auto_fallback:
            for name, config in self._runtimes.items():
                if api_key or name == "local":  # Skip hosted without API key
                    runtime = self._create_runtime_instance(name)
                    if runtime and runtime.is_available():
                        logger.info(f"Auto-detected runtime: {name}")
                        return runtime
        
        raise LLMRuntimeError(
            "No LLM runtime available. Please start mlx-llm-server locally or provide API keys for hosted services."
        )
    
    def _create_runtime_instance(self, name: str) -> Optional[LLMRuntime]:
        """Create a runtime instance by name."""
        if name not in self._runtimes:
            return None
        
        config = self._runtimes[name]
        try:
            return config["class"](**config["kwargs"])
        except Exception as e:
            logger.warning(f"Failed to create runtime '{name}': {e}")
            return None
    
    def list_available_runtimes(self) -> List[Dict[str, Any]]:
        """List all available runtimes with their status."""
        runtimes = []
        for name, config in self._runtimes.items():
            runtime = self._create_runtime_instance(name)
            if runtime:
                info = runtime.get_model_info()
                info["available"] = runtime.is_available()
                runtimes.append(info)
        return runtimes


class MockLLMRuntime:
    """Mock runtime for testing - returns predefined responses."""
    
    def __init__(self, responses: Dict[str, str]):
        """
        Args:
            responses: Map from prompt keywords to mock responses
        """
        self.responses = responses
        self.call_count = 0
        self.last_prompt = ""
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Return mock response based on prompt content."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Find matching response based on keywords
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return response
        
        # Default response if no match
        return '{"scenarios": [], "test_cases": [], "questions": []}'
    
    def is_available(self) -> bool:
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "mock",
            "type": "mock",
            "responses_count": len(self.responses)
        }


# Convenience functions for common use cases

def create_local_runtime(url: str = "http://localhost:8080/v1") -> LLMRuntime:
    """Create a local mlx-llm-server runtime."""
    return OpenAICompatibleRuntime(
        base_url=url,
        api_key="no-key", 
        name="mlx-llm-server"
    )


def create_openai_runtime(api_key: str, model: str = "gpt-3.5-turbo") -> LLMRuntime:
    """Create an OpenAI runtime."""
    if not api_key or api_key == "no-key":
        raise ConfigurationError("OpenAI API key is required")
    
    return OpenAICompatibleRuntime(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model=model,
        name="openai"
    )


def auto_detect_runtime(
    local_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMRuntime:
    """Auto-detect and create the best available runtime."""
    factory = RuntimeFactory()
    return factory.create_runtime(
        local_url=local_url,
        api_key=api_key,
        auto_fallback=True
    )