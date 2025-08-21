"""
JSON schema validation and retry logic for LLM responses.

Handles common LLM output issues: markdown wrapping, invalid JSON, missing fields.
Provides automatic repair and retry mechanisms for robust structured generation.
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from pydantic import BaseModel, ValidationError
import logging

from .runtime import LLMRuntime
from .exceptions import JSONValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class JSONValidator:
    """Validates and repairs LLM JSON responses with automatic retries."""
    
    def __init__(self, max_retries: int = 3, repair_attempts: int = 2):
        self.max_retries = max_retries
        self.repair_attempts = repair_attempts
    
    def validate_and_parse(
        self, 
        response: str, 
        model_class: Type[T],
        prompt_context: str = ""
    ) -> T:
        """
        Parse and validate LLM response into Pydantic model.
        
        Args:
            response: Raw LLM response
            model_class: Pydantic model class to parse into
            prompt_context: Context for better error messages
            
        Returns:
            Parsed and validated model instance
            
        Raises:
            JSONValidationError: If parsing fails after all repairs
        """
        
        # Step 1: Clean up common LLM formatting issues
        cleaned = self._clean_response(response)
        
        # Step 2: Attempt to parse JSON
        json_data = self._parse_json(cleaned)
        if json_data is None:
            # Try repair strategies
            for attempt in range(self.repair_attempts):
                logger.debug(f"JSON repair attempt {attempt + 1}/{self.repair_attempts}")
                repaired = self._repair_json(cleaned, attempt)
                json_data = self._parse_json(repaired)
                if json_data is not None:
                    break
        
        if json_data is None:
            logger.error(f"Failed to parse JSON from cleaned response: {cleaned[:500]}...")
            raise JSONValidationError(response, 1, 1)
        
        # Step 3: Validate against Pydantic model
        try:
            return model_class.parse_obj(json_data)
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed: {e}")
            # Try to repair common validation issues
            repaired_data = self._repair_validation_errors(json_data, e, model_class)
            if repaired_data:
                try:
                    return model_class.parse_obj(repaired_data)
                except ValidationError:
                    pass
            
            raise JSONValidationError(response, 1, 1)
    
    def validate_with_retries(
        self,
        runtime: LLMRuntime,
        prompt: str,
        model_class: Type[T],
        **generation_kwargs
    ) -> T:
        """
        Generate and validate response with automatic retries.
        
        Args:
            runtime: LLM runtime to use for generation
            prompt: Base prompt for generation
            model_class: Expected response model class
            **generation_kwargs: Additional arguments for runtime.generate()
            
        Returns:
            Parsed and validated model instance
            
        Raises:
            JSONValidationError: If all retries fail
        """
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Add schema instruction to prompt for clarity
                schema_prompt = self._enhance_prompt_with_schema(prompt, model_class)
                
                # Generate response
                response = runtime.generate(schema_prompt, **generation_kwargs)
                
                # Log the raw response for debugging
                logger.info(f"Raw LLM response (attempt {attempt}):")
                logger.info("=" * 80)
                logger.info(response)
                logger.info("=" * 80)
                
                # Validate and parse
                result = self.validate_and_parse(response, model_class, prompt[:100])
                
                if attempt > 1:
                    logger.info(f"Successfully parsed response on attempt {attempt}")
                
                return result
                
            except JSONValidationError as e:
                last_error = e
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                
                # Enhance prompt for next attempt
                if attempt < self.max_retries:
                    prompt = self._enhance_prompt_for_retry(prompt, e.raw_response, attempt)
        
        # All attempts failed
        raise JSONValidationError(
            last_error.raw_response if last_error else "",
            self.max_retries,
            self.max_retries
        )
    
    def _clean_response(self, response: str) -> str:
        """Clean common LLM formatting issues."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*\n?', '', response)
        response = re.sub(r'```\s*\n?', '', response)
        
        # Remove common prefixes/suffixes
        response = re.sub(r'^(Here\'s the.*?:|JSON:|Response:|schema)\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s*(That\'s the response|Hope this helps).*$', '', response, flags=re.IGNORECASE)
        
        # Strip whitespace
        response = response.strip()
        
        return response
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON, returning None on failure."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            logger.debug(f"Attempted to parse: {text[:200]}...")
            return None
    
    def _repair_json(self, text: str, attempt: int) -> str:
        """Apply repair strategies based on attempt number."""
        
        if attempt == 0:
            # Strategy 1: Find the scenarios array specifically (this is what we actually want)
            scenarios_match = re.search(r'"scenarios"\s*:\s*\[.*?\]', text, re.DOTALL)
            if scenarios_match:
                return "{" + scenarios_match.group(0) + "}"
            
            # Fallback: Find JSON-like content within text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group(0)
        
        elif attempt == 1:
            # Strategy 2: Fix common JSON issues
            # Add missing quotes around keys
            text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', text)
            
            # Fix trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # Fix single quotes
            text = text.replace("'", '"')
        
        return text
    
    def _repair_validation_errors(
        self, 
        data: Dict[str, Any], 
        error: ValidationError, 
        model_class: Type[BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to repair common Pydantic validation errors."""
        
        repaired = data.copy()
        
        for err in error.errors():
            field = err.get('loc', [None])[-1]  # Get the field name
            error_type = err.get('type', '')
            
            if error_type == 'value_error.missing':
                # Add missing required fields with sensible defaults
                if field and hasattr(model_class, '__fields__'):
                    field_info = model_class.__fields__.get(field)
                    if field_info:
                        if field_info.type_ == list:
                            repaired[field] = []
                        elif field_info.type_ == dict:
                            repaired[field] = {}
                        elif field_info.type_ == str:
                            repaired[field] = ""
                        elif field_info.type_ == bool:
                            repaired[field] = False
            
            elif error_type.startswith('type_error'):
                # Try to coerce types
                if field and field in repaired:
                    value = repaired[field]
                    if error_type == 'type_error.bool':
                        if isinstance(value, str):
                            repaired[field] = value.lower() in ('true', '1', 'yes')
                    elif error_type == 'type_error.integer':
                        try:
                            repaired[field] = int(float(str(value)))
                        except (ValueError, TypeError):
                            pass
        
        return repaired
    
    def _enhance_prompt_with_schema(self, prompt: str, model_class: Type[BaseModel]) -> str:
        """Add minimal schema guidance to prompt."""
        # For local models, don't include the full schema as it confuses them
        # The prompt should already have the example structure
        enhanced = f"""{prompt}

CRITICAL: Return ONLY valid JSON. No explanatory text, no markdown, no schema definitions."""
        
        return enhanced
    
    def _enhance_prompt_for_retry(self, prompt: str, failed_response: str, attempt: int) -> str:
        """Enhance prompt based on previous failure."""
        
        retry_guidance = f"""

RETRY ATTEMPT {attempt}: The previous response was invalid JSON. Common issues to avoid:
- Wrapping JSON in markdown code blocks (```json)
- Including explanatory text before/after JSON
- Missing quotes around string values
- Trailing commas
- Single quotes instead of double quotes

Previous failed response (for reference):
{failed_response[:200]}...

Please return ONLY valid JSON matching the schema."""
        
        return prompt + retry_guidance


# Convenience functions for common validation patterns

def validate_scenario_response(response: str) -> 'ScenarioGenerationResponse':
    """Validate scenario generation response."""
    from .models import ScenarioGenerationResponse
    validator = JSONValidator()
    return validator.validate_and_parse(response, ScenarioGenerationResponse)


def validate_test_case_response(response: str) -> 'TestCaseGenerationResponse':
    """Validate test case generation response."""
    from .models import TestCaseGenerationResponse
    validator = JSONValidator()
    return validator.validate_and_parse(response, TestCaseGenerationResponse)


def generate_with_validation(
    runtime: LLMRuntime,
    prompt: str,
    response_model: Type[T],
    max_retries: int = 3,
    **generation_kwargs
) -> T:
    """
    Generate and validate LLM response in one call.
    
    This is the main function that combines generation and validation
    with automatic retries for robust JSON output.
    """
    validator = JSONValidator(max_retries=max_retries)
    return validator.validate_with_retries(
        runtime, prompt, response_model, **generation_kwargs
    )