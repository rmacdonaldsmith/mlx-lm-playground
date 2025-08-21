"""
ParseRequirements Node (Deterministic)

Normalizes acceptance criteria to stable IDs and extracts key entities/fields.
This is the only fully deterministic node in the workflow.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from ..models import (
    RequirementsInput, 
    ParsedRequirements, 
    AcceptanceCriteria, 
    Constraints
)


@dataclass
class EntityExtractionResult:
    """Results from entity extraction."""
    fields: Set[str]          # Form fields, database columns, etc.
    actions: Set[str]         # User actions (click, submit, validate, etc.)
    data_types: Set[str]      # Data types (email, phone, ZIP, credit card, etc.)
    validations: Set[str]     # Validation rules (required, format, length, etc.)
    error_conditions: Set[str] # Error conditions (invalid format, missing data, etc.)
    business_rules: Set[str]   # Business logic (permissions, calculations, etc.)


class ParseRequirements:
    """
    Node 1: Parse and normalize requirements input.
    
    This node is purely deterministic - given the same input, it always produces
    the same output. No LLM calls are made here.
    """
    
    @staticmethod
    def process(requirements: RequirementsInput) -> ParsedRequirements:
        """
        Process requirements input and return normalized structure.
        
        Args:
            requirements: Raw requirements input
            
        Returns:
            ParsedRequirements with normalized ACs and extracted entities
        """
        
        # Step 1: Normalize acceptance criteria to stable IDs
        normalized_acs = ParseRequirements._normalize_acceptance_criteria(
            requirements.acceptance_criteria
        )
        
        # Step 2: Extract entities from spec text and ACs for context
        all_text = requirements.spec_text + " " + " ".join(requirements.acceptance_criteria)
        entities = ParseRequirements._extract_entities(all_text)
        
        # Step 3: Carry forward constraints
        constraints = requirements.constraints
        
        return ParsedRequirements(
            acceptance_criteria=normalized_acs,
            entities=entities,
            constraints=constraints
        )
    
    @staticmethod
    def _normalize_acceptance_criteria(criteria: List[str]) -> List[AcceptanceCriteria]:
        """
        Convert list of AC strings to normalized AcceptanceCriteria objects.
        
        Uses deterministic ordering - same input always produces same IDs.
        """
        normalized = []
        
        for i, criterion in enumerate(criteria, 1):
            # Generate stable ID: AC1, AC2, AC3, etc.
            ac_id = f"AC{i}"
            
            # Clean up the criterion text
            cleaned_text = criterion.strip()
            if not cleaned_text:
                continue
                
            normalized.append(AcceptanceCriteria(
                id=ac_id,
                text=cleaned_text
            ))
        
        return normalized
    
    @staticmethod
    def _extract_entities(text: str) -> Dict[str, Any]:
        """
        Extract key entities and patterns from requirements text.
        
        This provides context hints for LLM nodes but doesn't drive generation.
        Uses simple regex patterns - more sophisticated NLP could be added later.
        """
        
        text_lower = text.lower()
        
        # Extract different types of entities
        fields = ParseRequirements._extract_fields(text_lower)
        actions = ParseRequirements._extract_actions(text_lower)
        data_types = ParseRequirements._extract_data_types(text_lower)
        validations = ParseRequirements._extract_validations(text_lower)
        error_conditions = ParseRequirements._extract_error_conditions(text_lower)
        business_rules = ParseRequirements._extract_business_rules(text_lower)
        
        # Build summary for LLM context
        extraction_result = EntityExtractionResult(
            fields=fields,
            actions=actions,
            data_types=data_types,
            validations=validations,
            error_conditions=error_conditions,
            business_rules=business_rules
        )
        
        return {
            "fields": sorted(list(fields)),
            "actions": sorted(list(actions)),
            "data_types": sorted(list(data_types)),
            "validations": sorted(list(validations)),
            "error_conditions": sorted(list(error_conditions)),
            "business_rules": sorted(list(business_rules)),
            "summary": ParseRequirements._create_entity_summary(extraction_result)
        }
    
    @staticmethod
    def _extract_fields(text: str) -> Set[str]:
        """Extract form fields, database columns, UI elements."""
        patterns = [
            r'\b(?:field|input|textbox|dropdown|checkbox|button|form)\b',
            r'\b(?:name|email|phone|address|city|state|zip|postal)\b',
            r'\b(?:card|number|cvv|expiry|expiration)\b',
            r'\b(?:password|username|login|account)\b',
            r'\b(?:first|last|middle)\s+name\b',
            r'\b(?:billing|shipping)\s+(?:address|info)\b',
        ]
        
        fields = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            fields.update(matches)
        
        return fields
    
    @staticmethod
    def _extract_actions(text: str) -> Set[str]:
        """Extract user actions and system behaviors."""
        patterns = [
            r'\b(?:click|tap|press|select|choose|enter|type|input)\b',
            r'\b(?:submit|save|cancel|delete|update|create|add|remove)\b',
            r'\b(?:validate|verify|check|confirm|approve|reject)\b',
            r'\b(?:login|logout|signin|signout|register)\b',
            r'\b(?:redirect|navigate|display|show|hide|open|close)\b',
        ]
        
        actions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            actions.update(matches)
        
        return actions
    
    @staticmethod
    def _extract_data_types(text: str) -> Set[str]:
        """Extract data types and formats."""
        patterns = [
            r'\b(?:email|phone|zip|postal|ssn|credit card|debit card)\b',
            r'\b(?:date|time|datetime|timestamp)\b',
            r'\b(?:currency|money|price|amount|dollar|usd)\b',
            r'\b(?:url|link|website|domain)\b',
            r'\b(?:number|integer|float|decimal|percentage)\b',
            r'\b(?:string|text|varchar|char)\b',
            r'\b(?:boolean|true|false|yes|no)\b',
        ]
        
        data_types = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            data_types.update(matches)
        
        return data_types
    
    @staticmethod
    def _extract_validations(text: str) -> Set[str]:
        """Extract validation rules and requirements."""
        patterns = [
            r'\b(?:required|mandatory|optional|nullable)\b',
            r'\b(?:minimum|maximum|min|max|length|size)\b',
            r'\b(?:format|pattern|regex|match|valid|invalid)\b',
            r'\b(?:unique|duplicate|exists|available)\b',
            r'\bstronger?\s+(?:password|auth)\b',
            r'\bpass(?:es)?\s+(?:luhn|checksum|validation)\b',
        ]
        
        validations = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            validations.update(matches)
        
        return validations
    
    @staticmethod
    def _extract_error_conditions(text: str) -> Set[str]:
        """Extract error conditions and failure scenarios."""
        patterns = [
            r'\b(?:error|fail|failure|invalid|incorrect|wrong)\b',
            r'\b(?:missing|empty|null|blank|undefined)\b',
            r'\b(?:timeout|expired|limit|exceeded|too\s+(?:long|short|many))\b',
            r'\b(?:unauthorized|forbidden|denied|blocked)\b',
            r'\b(?:connection|network|server)\s+(?:error|fail|down)\b',
        ]
        
        error_conditions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            error_conditions.update(matches)
        
        return error_conditions
    
    @staticmethod
    def _extract_business_rules(text: str) -> Set[str]:
        """Extract business logic and domain rules."""
        patterns = [
            r'\b(?:business|domain|company|organization)\s+(?:rule|logic|policy)\b',
            r'\b(?:permission|access|role|admin|user|guest)\b',
            r'\b(?:payment|billing|subscription|plan|tier)\b',
            r'\b(?:discount|coupon|promo|offer|sale)\b',
            r'\b(?:inventory|stock|available|sold out)\b',
            r'\b(?:shipping|delivery|fulfillment)\b',
            r'\b(?:tax|vat|gst|fee|charge|cost)\b',
        ]
        
        business_rules = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            business_rules.update(matches)
        
        return business_rules
    
    @staticmethod
    def _create_entity_summary(result: EntityExtractionResult) -> str:
        """Create a human-readable summary of extracted entities."""
        summary_parts = []
        
        if result.fields:
            fields_str = ", ".join(sorted(list(result.fields))[:5])
            summary_parts.append(f"Key fields: {fields_str}")
        
        if result.data_types:
            types_str = ", ".join(sorted(list(result.data_types))[:3])
            summary_parts.append(f"Data types: {types_str}")
        
        if result.validations:
            validations_str = ", ".join(sorted(list(result.validations))[:3])
            summary_parts.append(f"Validations: {validations_str}")
        
        if result.error_conditions:
            errors_str = ", ".join(sorted(list(result.error_conditions))[:3])
            summary_parts.append(f"Error conditions: {errors_str}")
        
        return ". ".join(summary_parts) if summary_parts else "No specific patterns detected."


# Convenience functions for common parsing tasks

def parse_requirements(requirements: RequirementsInput) -> ParsedRequirements:
    """Convenience function to parse requirements."""
    return ParseRequirements.process(requirements)


def normalize_ac_list(criteria: List[str]) -> List[AcceptanceCriteria]:
    """Convenience function to normalize AC list."""
    return ParseRequirements._normalize_acceptance_criteria(criteria)