"""Pytest configuration and fixtures for QA generator tests."""

import pytest
from typing import Dict, Any, List
from qa_generator.runtime import MockLLMRuntime
from qa_generator.models import RequirementsInput, Constraints, AcceptanceCriteria


@pytest.fixture
def mock_runtime():
    """Create mock LLM runtime with predefined responses."""
    responses = {
        "scenario": """
        {
          "scenarios": [
            {
              "id": "SCN-001",
              "title": "Valid ZIP code provided",
              "type": "functional",
              "risk": "medium", 
              "related_requirements": ["AC1"],
              "preconditions": "User is on payment form",
              "variants": null
            },
            {
              "id": "SCN-002",
              "title": "Missing ZIP code",
              "type": "functional",
              "risk": "medium",
              "related_requirements": ["AC1"],
              "preconditions": "User is on payment form",
              "variants": null
            },
            {
              "id": "SCN-003", 
              "title": "Valid Luhn credit card",
              "type": "functional",
              "risk": "high",
              "related_requirements": ["AC2"],
              "preconditions": "User enters card details",
              "variants": null
            },
            {
              "id": "SCN-004",
              "title": "Invalid Luhn credit card",
              "type": "functional",
              "risk": "high",
              "related_requirements": ["AC2"],
              "preconditions": "User enters card details", 
              "variants": null
            }
          ]
        }
        """,
        
        "test_case": """
        {
          "test_cases": [
            {
              "id": "TC-001",
              "scenario_id": "SCN-001",
              "case_type": "functional",
              "priority": "P1",
              "steps": [
                "Navigate to payment form",
                "Enter valid ZIP code '12345'",
                "Click Submit button"
              ],
              "data": {"zip_code": "12345"},
              "expected": [
                "Form submits successfully",
                "No validation errors shown"
              ],
              "negative": false,
              "tags": ["zip", "validation"]
            },
            {
              "id": "TC-002", 
              "scenario_id": "SCN-002",
              "case_type": "negative",
              "priority": "P1",
              "steps": [
                "Navigate to payment form",
                "Leave ZIP code field empty",
                "Click Submit button"
              ],
              "data": {"zip_code": ""},
              "expected": [
                "Form does not submit",
                "Error message: 'ZIP code is required'"
              ],
              "negative": true,
              "tags": ["zip", "validation", "error"]
            }
          ]
        }
        """
    }
    
    return MockLLMRuntime(responses)


@pytest.fixture
def sample_requirements():
    """Create sample requirements for testing."""
    return RequirementsInput(
        project="payment-flow",
        artifact_id="JIRA-123",
        spec_text="Payment form for credit card processing with ZIP code validation and Luhn algorithm verification.",
        acceptance_criteria=[
            "US ZIP code is required when saving a card",
            "Card number must pass Luhn validation; otherwise show a validation error"
        ],
        constraints=Constraints(
            test_framework="playwright",
            priority_policy="risk_weighted"
        )
    )


@pytest.fixture
def sample_normalized_acs():
    """Create sample normalized acceptance criteria."""
    return [
        AcceptanceCriteria(id="AC1", text="US ZIP code is required when saving a card"),
        AcceptanceCriteria(id="AC2", text="Card number must pass Luhn validation; otherwise show a validation error")
    ]


@pytest.fixture  
def g1_violation_scenarios():
    """Create scenarios that would violate G1 gates for testing."""
    from qa_generator.models import Scenario, TestCase
    
    # Scenarios that don't cover all ACs (G1.1 violation)
    incomplete_scenarios = [
        Scenario(
            id="SCN-001",
            title="Only covers AC1",
            type="functional", 
            risk="medium",
            related_requirements=["AC1"],
            preconditions=None,
            variants=None
        )
        # Missing scenario for AC2 -> G1.1 violation
    ]
    
    # Test cases that don't have negative coverage (G1.3 violation) 
    incomplete_test_cases = [
        TestCase(
            id="TC-001",
            scenario_id="SCN-001",
            case_type="functional",
            priority="P1",
            steps=["Test step"],
            data=None,
            expected=["Expected result"],
            negative=False,  # Only positive, no negative -> G1.3 violation
            tags=None
        )
    ]
    
    return {
        "scenarios": incomplete_scenarios,
        "test_cases": incomplete_test_cases
    }


@pytest.fixture
def zip_luhn_example():
    """Create the ZIP/Luhn example from the requirements for testing."""
    return {
        "project": "payment-system",
        "artifact_id": "EXAMPLE-001",
        "spec_text": """
        Payment processing system that accepts credit cards.
        The system must validate ZIP codes for US addresses and 
        verify credit card numbers using the Luhn algorithm.
        """,
        "acceptance_criteria": [
            "US ZIP code is required when saving a card",
            "Card number must pass Luhn; otherwise show a validation error"
        ]
    }