"""
Simple MLX Runtime - Direct in-process generation.

KISS principle: Just use MLX directly without HTTP complexity.
Clean, simple, and works reliably.
"""

from typing import Dict, Any
import logging
from .exceptions import LLMRuntimeError

logger = logging.getLogger(__name__)


class SimpleMLXRuntime:
    """
    Simple runtime that uses MLX directly in-process.
    No HTTP, no complexity - just works.
    """
    
    def __init__(self, model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"):
        self.model_path = model_path
        self.name = "simple-mlx"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model once on initialization."""
        try:
            from mlx_lm import load
            logger.info(f"Loading MLX model: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path)
            logger.info("MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise LLMRuntimeError(f"Could not load MLX model: {e}")
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """Generate response using MLX directly."""
        
        if not self.model or not self.tokenizer:
            raise LLMRuntimeError("MLX model not loaded")
        
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            
            # Create sampler with temperature and top_p
            sampler = make_sampler(temp=temperature, top_p=top_p)
            
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                sampler=sampler,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise LLMRuntimeError(f"MLX generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx_lm
            return self.model is not None and self.tokenizer is not None
        except ImportError:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "type": "simple-mlx", 
            "model_path": self.model_path,
            "loaded": self.model is not None
        }