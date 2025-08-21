"""Custom exceptions for QA generator."""

from typing import List, Optional


class QAGeneratorError(Exception):
    """Base exception for QA generator errors."""
    pass


class G1ValidationError(QAGeneratorError):
    """
    Raised when generated artifacts violate quality gates G1.1-G1.5.
    
    Attributes:
        violated_rules: List of rule IDs that failed (e.g., ['G1.1', 'G1.3'])
        offending_ids: IDs of ACs, scenarios, or test cases causing the failure
        details: Human-readable explanation of what went wrong
    """
    
    def __init__(self, violated_rules: List[str], offending_ids: List[str], details: str):
        self.violated_rules = violated_rules
        self.offending_ids = offending_ids 
        self.details = details
        super().__init__(f"G1 validation failed: {', '.join(violated_rules)}. {details}")


class JSONValidationError(QAGeneratorError):
    """Raised when LLM returns invalid JSON after maximum retries."""
    
    def __init__(self, raw_response: str, attempts: int, max_retries: int):
        self.raw_response = raw_response
        self.attempts = attempts
        self.max_retries = max_retries
        super().__init__(f"Failed to parse valid JSON after {attempts}/{max_retries} attempts")


class LLMRuntimeError(QAGeneratorError):
    """Raised when LLM runtime encounters an error."""
    pass


class ConfigurationError(QAGeneratorError):
    """Raised when configuration is invalid or missing."""
    pass