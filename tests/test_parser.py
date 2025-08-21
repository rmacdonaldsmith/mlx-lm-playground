"""Test ParseRequirements node (deterministic parsing)."""

import pytest
from qa_generator.nodes.parser import ParseRequirements
from qa_generator.models import RequirementsInput, AcceptanceCriteria


class TestParseRequirements:
    """Test the deterministic ParseRequirements node."""
    
    def test_normalize_acceptance_criteria(self):
        """Test AC normalization to stable IDs."""
        criteria = [
            "ZIP code is required",
            "Email must be valid format", 
            "Password must be at least 8 characters"
        ]
        
        normalized = ParseRequirements._normalize_acceptance_criteria(criteria)
        
        assert len(normalized) == 3
        assert normalized[0].id == "AC1"
        assert normalized[0].text == "ZIP code is required"
        assert normalized[1].id == "AC2"
        assert normalized[1].text == "Email must be valid format"
        assert normalized[2].id == "AC3" 
        assert normalized[2].text == "Password must be at least 8 characters"
    
    def test_normalize_empty_criteria_filtered(self):
        """Test that empty criteria are filtered out."""
        criteria = [
            "Valid criterion",
            "",  # Empty
            "  ",  # Whitespace only
            "Another valid criterion"
        ]
        
        normalized = ParseRequirements._normalize_acceptance_criteria(criteria)
        
        assert len(normalized) == 2
        assert normalized[0].id == "AC1"
        assert normalized[0].text == "Valid criterion"
        assert normalized[1].id == "AC2"
        assert normalized[1].text == "Another valid criterion"
    
    def test_extract_entities_fields(self):
        """Test field extraction from requirements text."""
        text = "The user must enter their email address and phone number in the contact form."
        
        entities = ParseRequirements._extract_entities(text)
        
        assert "fields" in entities
        fields = entities["fields"]
        assert "email" in fields
        assert "phone" in fields
        assert "form" in fields
    
    def test_extract_entities_actions(self):
        """Test action extraction."""
        text = "User clicks submit button, then system validates and redirects to confirmation page."
        
        entities = ParseRequirements._extract_entities(text)
        
        assert "actions" in entities
        actions = entities["actions"]
        assert "click" in actions
        assert "submit" in actions
        assert "validate" in actions
        assert "redirect" in actions
    
    def test_extract_entities_data_types(self):
        """Test data type extraction."""
        text = "Accept email addresses, phone numbers, ZIP codes, and credit card numbers with CVV."
        
        entities = ParseRequirements._extract_entities(text)
        
        assert "data_types" in entities
        data_types = entities["data_types"]
        assert "email" in data_types
        assert "phone" in data_types
        assert "zip" in data_types
        assert "credit card" in data_types
    
    def test_extract_entities_validations(self):
        """Test validation rule extraction."""
        text = "Email field is required and must match valid format. Password minimum length is 8 characters."
        
        entities = ParseRequirements._extract_entities(text)
        
        assert "validations" in entities
        validations = entities["validations"]
        assert "required" in validations
        assert "valid" in validations
        assert "format" in validations
        assert "minimum" in validations
    
    def test_extract_entities_error_conditions(self):
        """Test error condition extraction."""
        text = "Show error message if validation fails or network timeout occurs."
        
        entities = ParseRequirements._extract_entities(text)
        
        assert "error_conditions" in entities
        errors = entities["error_conditions"]
        assert "error" in errors
        assert "fail" in errors
        assert "timeout" in errors
    
    def test_process_complete_workflow(self, sample_requirements):
        """Test complete ParseRequirements processing."""
        parsed = ParseRequirements.process(sample_requirements)
        
        # Check normalized ACs
        assert len(parsed.acceptance_criteria) == 2
        assert parsed.acceptance_criteria[0].id == "AC1"
        assert parsed.acceptance_criteria[1].id == "AC2"
        assert "ZIP code" in parsed.acceptance_criteria[0].text
        assert "Luhn" in parsed.acceptance_criteria[1].text
        
        # Check entity extraction
        assert parsed.entities is not None
        assert "fields" in parsed.entities
        assert "validations" in parsed.entities
        assert "summary" in parsed.entities
        
        # Check constraints passed through
        assert parsed.constraints is not None
        assert parsed.constraints.test_framework == "playwright"
        assert parsed.constraints.priority_policy == "risk_weighted"
    
    def test_deterministic_output(self):
        """Test that same input always produces same output."""
        requirements = RequirementsInput(
            project="test",
            artifact_id="TEST-1",
            spec_text="User must enter valid email and phone number.",
            acceptance_criteria=["Email is required", "Phone is optional"]
        )
        
        # Process twice
        result1 = ParseRequirements.process(requirements)
        result2 = ParseRequirements.process(requirements)
        
        # Results should be identical
        assert result1.acceptance_criteria == result2.acceptance_criteria
        assert result1.entities == result2.entities
        assert result1.constraints == result2.constraints
    
    def test_entity_extraction_edge_cases(self):
        """Test entity extraction with edge cases."""
        # Empty text
        entities_empty = ParseRequirements._extract_entities("")
        assert all(len(entities_empty[key]) == 0 for key in ["fields", "actions", "data_types"])
        
        # Text without relevant patterns
        entities_plain = ParseRequirements._extract_entities("This is just plain text without technical terms.")
        assert "summary" in entities_plain
        
        # Mixed case and special characters
        text_mixed = "User CLICKS the Submit-Button to VALIDATE Email_Address@domain.com!"
        entities_mixed = ParseRequirements._extract_entities(text_mixed)
        
        # Should still extract patterns (case-insensitive)
        assert "click" in entities_mixed["actions"]
        assert "submit" in entities_mixed["actions"] 
        assert "validate" in entities_mixed["actions"]
        assert "email" in entities_mixed["data_types"]


class TestEntityExtraction:
    """Test individual entity extraction methods."""
    
    def test_extract_fields_comprehensive(self):
        """Test comprehensive field extraction."""
        text = "registration form with name field, email input, password textbox, and submit button"
        
        fields = ParseRequirements._extract_fields(text)
        
        expected_fields = {"form", "field", "name", "email", "password", "button", "input", "textbox"}
        assert expected_fields.issubset(set(fields))
    
    def test_extract_actions_comprehensive(self):
        """Test comprehensive action extraction."""
        text = "user clicks login, enters credentials, submits form, system validates and redirects"
        
        actions = ParseRequirements._extract_actions(text)
        
        expected_actions = {"click", "enter", "submit", "validate", "redirect", "login"}
        assert expected_actions.issubset(set(actions))
    
    def test_extract_business_rules(self):
        """Test business rule extraction."""
        text = "admin users have special permissions for payment processing and billing operations"
        
        business_rules = ParseRequirements._extract_business_rules(text)
        
        assert "permission" in business_rules
        assert "admin" in business_rules
        assert "payment" in business_rules
        assert "billing" in business_rules
    
    def test_create_entity_summary(self):
        """Test entity summary creation."""
        from qa_generator.nodes.parser import EntityExtractionResult
        
        result = EntityExtractionResult(
            fields={"email", "password", "name"},
            actions={"click", "submit", "validate"},
            data_types={"email", "password"},
            validations={"required", "format"},
            error_conditions={"invalid", "missing"},
            business_rules={"permission", "access"}
        )
        
        summary = ParseRequirements._create_entity_summary(result)
        
        assert "email" in summary
        assert "password" in summary
        assert "required" in summary
        assert len(summary) > 0
        assert isinstance(summary, str)