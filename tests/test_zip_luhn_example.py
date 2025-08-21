"""Test the ZIP code/Luhn validation example from requirements."""

import pytest
from unittest.mock import patch
import json

from qa_generator.workflow import QAWorkflow
from qa_generator.models import RequirementsInput, Constraints
from qa_generator.runtime import MockLLMRuntime
from qa_generator.exceptions import G1ValidationError


class TestZipLuhnExample:
    """Test the example provided in requirements: ZIP code and Luhn validation."""
    
    @pytest.fixture
    def zip_luhn_requirements(self):
        """Create ZIP/Luhn requirements as specified in the original requirements."""
        return RequirementsInput(
            project="payment-system",
            artifact_id="EXAMPLE-001",
            spec_text="""
            Payment processing system that accepts credit cards.
            The system must validate ZIP codes for US addresses and 
            verify credit card numbers using the Luhn algorithm.
            Error messages should be shown to users when validation fails.
            """,
            acceptance_criteria=[
                "US ZIP code is required when saving a card",
                "Card number must pass Luhn; otherwise show a validation error"
            ],
            constraints=Constraints(
                test_framework="playwright",
                priority_policy="risk_weighted"
            )
        )
    
    @pytest.fixture
    def zip_luhn_mock_runtime(self):
        """Mock runtime with realistic ZIP/Luhn responses."""
        responses = {
            "scenario": """
            {
              "scenarios": [
                {
                  "id": "SCN-001",
                  "title": "Valid US ZIP code provided when saving card",
                  "type": "functional",
                  "risk": "medium",
                  "related_requirements": ["AC1"],
                  "preconditions": "User is on card registration form",
                  "variants": ["5-digit", "9-digit ZIP+4"]
                },
                {
                  "id": "SCN-002", 
                  "title": "Missing ZIP code when saving card",
                  "type": "functional",
                  "risk": "medium",
                  "related_requirements": ["AC1"],
                  "preconditions": "User is on card registration form",
                  "variants": ["empty field", "whitespace only"]
                },
                {
                  "id": "SCN-003",
                  "title": "Invalid ZIP code format", 
                  "type": "functional",
                  "risk": "medium",
                  "related_requirements": ["AC1"],
                  "preconditions": "User is on card registration form",
                  "variants": ["too short", "too long", "non-numeric"]
                },
                {
                  "id": "SCN-004",
                  "title": "Valid credit card number passes Luhn check",
                  "type": "functional", 
                  "risk": "high",
                  "related_requirements": ["AC2"],
                  "preconditions": "User enters card details",
                  "variants": ["Visa", "Mastercard", "Amex"]
                },
                {
                  "id": "SCN-005",
                  "title": "Invalid credit card number fails Luhn check",
                  "type": "functional",
                  "risk": "high", 
                  "related_requirements": ["AC2"],
                  "preconditions": "User enters card details",
                  "variants": ["wrong checksum", "invalid format"]
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
                    "Navigate to card registration form",
                    "Enter valid card number: 4532015112830366",
                    "Enter valid ZIP code: 12345",
                    "Click Save Card button"
                  ],
                  "data": {
                    "card_number": "4532015112830366",
                    "zip_code": "12345"
                  },
                  "expected": [
                    "Card is saved successfully",
                    "Success confirmation message displayed",
                    "No validation errors shown"
                  ],
                  "negative": false,
                  "tags": ["zip", "positive", "validation"]
                },
                {
                  "id": "TC-002",
                  "scenario_id": "SCN-002", 
                  "case_type": "negative",
                  "priority": "P1",
                  "steps": [
                    "Navigate to card registration form",
                    "Enter valid card number: 4532015112830366", 
                    "Leave ZIP code field empty",
                    "Click Save Card button"
                  ],
                  "data": {
                    "card_number": "4532015112830366",
                    "zip_code": ""
                  },
                  "expected": [
                    "Card is not saved",
                    "Error message: 'ZIP code is required'",
                    "ZIP code field highlighted in red"
                  ],
                  "negative": true,
                  "tags": ["zip", "negative", "required", "error"]
                },
                {
                  "id": "TC-003",
                  "scenario_id": "SCN-004",
                  "case_type": "functional",
                  "priority": "P0",
                  "steps": [
                    "Navigate to card registration form", 
                    "Enter valid Luhn card number: 4532015112830366",
                    "Enter valid ZIP code: 90210",
                    "Click Save Card button"
                  ],
                  "data": {
                    "card_number": "4532015112830366",
                    "zip_code": "90210"
                  },
                  "expected": [
                    "Luhn validation passes",
                    "Card is saved successfully",
                    "No error messages shown"
                  ],
                  "negative": false,
                  "tags": ["luhn", "positive", "card", "validation"]
                },
                {
                  "id": "TC-004",
                  "scenario_id": "SCN-005",
                  "case_type": "negative", 
                  "priority": "P0",
                  "steps": [
                    "Navigate to card registration form",
                    "Enter invalid Luhn card number: 1234567890123456",
                    "Enter valid ZIP code: 90210", 
                    "Click Save Card button"
                  ],
                  "data": {
                    "card_number": "1234567890123456",
                    "zip_code": "90210"
                  },
                  "expected": [
                    "Luhn validation fails", 
                    "Error message: 'Invalid card number'",
                    "Card number field highlighted in red",
                    "Card is not saved"
                  ],
                  "negative": true,
                  "tags": ["luhn", "negative", "card", "error", "validation"]
                }
              ]
            }
            """
        }
        
        return MockLLMRuntime(responses)
    
    def test_zip_luhn_complete_workflow(self, zip_luhn_requirements, zip_luhn_mock_runtime):
        """Test complete workflow with ZIP/Luhn example."""
        workflow = QAWorkflow(runtime=zip_luhn_mock_runtime)
        
        results = workflow.run(zip_luhn_requirements)
        
        # Validate input processing
        assert results["input"].project == "payment-system"
        assert results["input"].artifact_id == "EXAMPLE-001"
        
        # Validate parsed requirements
        parsed = results["parsed_requirements"]
        assert len(parsed.acceptance_criteria) == 2
        assert parsed.acceptance_criteria[0].id == "AC1"
        assert parsed.acceptance_criteria[1].id == "AC2"
        assert "ZIP code" in parsed.acceptance_criteria[0].text
        assert "Luhn" in parsed.acceptance_criteria[1].text
        
        # Validate scenarios
        scenarios = results["scenarios"]
        assert len(scenarios) == 5
        
        # Check scenario types and coverage
        scenario_titles = [s.title for s in scenarios]
        assert any("Valid US ZIP" in title for title in scenario_titles)
        assert any("Missing ZIP" in title for title in scenario_titles)
        assert any("Invalid ZIP" in title for title in scenario_titles)
        assert any("Valid credit card" in title for title in scenario_titles)
        assert any("Invalid credit card" in title for title in scenario_titles)
        
        # Validate test cases
        test_cases = results["test_cases"]
        assert len(test_cases) == 4
        
        # Check positive and negative coverage
        positive_cases = [tc for tc in test_cases if not tc.negative]
        negative_cases = [tc for tc in test_cases if tc.negative]
        assert len(positive_cases) >= 2
        assert len(negative_cases) >= 2
        
        # Validate coverage map
        coverage_map = results["coverage_map"]
        assert "AC1" in coverage_map.ac_to_scenarios
        assert "AC2" in coverage_map.ac_to_scenarios
        assert len(coverage_map.ac_to_scenarios["AC1"]) >= 1
        assert len(coverage_map.ac_to_scenarios["AC2"]) >= 1
        
        # Validate open questions
        open_questions = results["open_questions"]
        assert len(open_questions) > 0
        
        # Should have questions about ZIP formats and international postal codes
        question_texts = [q.text for q in open_questions]
        zip_questions = [q for q in question_texts if "ZIP" in q or "postal" in q]
        assert len(zip_questions) > 0
        
        # Validate artifacts
        artifacts = results["artifacts"]
        assert "json_plan_path" in artifacts
        assert "test_plan" in artifacts
        
        # Validate execution stats
        stats = results["execution_stats"]
        assert stats["node1_acs"] == 2
        assert stats["node2_scenarios"] == 5
        assert stats["node3_test_cases"] == 4
        assert stats["node4_coverage"] == 2
    
    def test_zip_luhn_g1_compliance(self, zip_luhn_requirements, zip_luhn_mock_runtime):
        """Test that ZIP/Luhn example passes all G1 quality gates."""
        workflow = QAWorkflow(runtime=zip_luhn_mock_runtime)
        
        # Should not raise G1ValidationError
        results = workflow.run(zip_luhn_requirements)
        
        # Manually verify G1 compliance
        acs = results["parsed_requirements"].acceptance_criteria
        scenarios = results["scenarios"]
        test_cases = results["test_cases"]
        
        from qa_generator.nodes.critic import G1Validator
        
        # Should pass all G1 gates
        G1Validator.validate_all_gates(acs, scenarios, test_cases)
        
        # Validate specific G1 requirements
        
        # G1.1: Every AC has scenario coverage
        coverage_map = results["coverage_map"]
        for ac in acs:
            assert ac.id in coverage_map.ac_to_scenarios
            assert len(coverage_map.ac_to_scenarios[ac.id]) >= 1
        
        # G1.2: Every scenario has test cases  
        scenario_ids = {s.id for s in scenarios}
        covered_scenario_ids = {tc.scenario_id for tc in test_cases}
        assert scenario_ids.issubset(covered_scenario_ids)
        
        # G1.3: Each AC has positive AND negative test cases
        from qa_generator.nodes.critic import G1Validator
        ac_to_test_cases = G1Validator._build_ac_to_test_cases_map(scenarios, test_cases)
        
        for ac in acs:
            ac_test_cases = ac_to_test_cases[ac.id]
            positive_cases = [tc for tc in ac_test_cases if not tc.negative]
            negative_cases = [tc for tc in ac_test_cases if tc.negative]
            
            assert len(positive_cases) >= 1, f"AC {ac.id} missing positive test cases"
            assert len(negative_cases) >= 1, f"AC {ac.id} missing negative test cases"
    
    def test_zip_luhn_priority_assignment(self, zip_luhn_requirements, zip_luhn_mock_runtime):
        """Test that ZIP/Luhn example has proper priority assignment.""" 
        workflow = QAWorkflow(runtime=zip_luhn_mock_runtime)
        results = workflow.run(zip_luhn_requirements)
        
        test_cases = results["test_cases"]
        
        # Check priority distribution
        priorities = [tc.priority for tc in test_cases]
        
        # Should have P0 cases (high-risk payment scenarios)
        assert "P0" in priorities
        
        # Should have P1 cases (important functionality)
        assert "P1" in priorities
        
        # Luhn validation should be high priority (P0) due to high risk
        luhn_cases = [tc for tc in test_cases if any("luhn" in tag.lower() for tag in (tc.tags or []))]
        if luhn_cases:
            assert all(tc.priority in ["P0", "P1"] for tc in luhn_cases)
    
    def test_zip_luhn_test_data_validity(self, zip_luhn_requirements, zip_luhn_mock_runtime):
        """Test that generated test data is realistic and valid."""
        workflow = QAWorkflow(runtime=zip_luhn_mock_runtime)
        results = workflow.run(zip_luhn_requirements)
        
        test_cases = results["test_cases"]
        
        # Check test data structure
        cases_with_data = [tc for tc in test_cases if tc.data]
        assert len(cases_with_data) > 0
        
        for tc in cases_with_data:
            data = tc.data
            
            # ZIP code validation
            if "zip_code" in data:
                zip_code = data["zip_code"]
                if zip_code:  # Non-empty ZIP codes should be valid format
                    assert isinstance(zip_code, str)
                    # Should be 5 digits for valid cases
                    if not tc.negative and len(zip_code) == 5:
                        assert zip_code.isdigit()
            
            # Card number validation  
            if "card_number" in data:
                card_number = data["card_number"]
                assert isinstance(card_number, str)
                # Should be numeric string
                assert card_number.replace(" ", "").isdigit()
    
    def test_zip_luhn_open_questions_quality(self, zip_luhn_requirements, zip_luhn_mock_runtime):
        """Test that open questions are relevant and well-formed."""
        workflow = QAWorkflow(runtime=zip_luhn_mock_runtime)
        results = workflow.run(zip_luhn_requirements)
        
        open_questions = results["open_questions"]
        
        # Should have relevant questions
        assert len(open_questions) > 0
        
        # Validate question structure
        for question in open_questions:
            assert question.id.startswith("Q-")
            assert len(question.text) > 0
            assert isinstance(question.blocking, bool)
            
            # Question should be relevant to ZIP/Luhn domain
            text_lower = question.text.lower()
            relevant_terms = ["zip", "postal", "card", "luhn", "validation", "format", "error"]
            assert any(term in text_lower for term in relevant_terms)
        
        # Should have questions about international postal codes
        postal_questions = [q for q in open_questions if "postal" in q.text.lower() or "international" in q.text.lower()]
        assert len(postal_questions) > 0
        
        # Blocking vs non-blocking questions should be appropriate
        blocking_questions = [q for q in open_questions if q.blocking]
        non_blocking_questions = [q for q in open_questions if not q.blocking]
        
        # Should have both types
        assert len(blocking_questions) >= 0  # May have blocking questions
        assert len(non_blocking_questions) >= 1  # Should have non-blocking questions


@pytest.mark.integration
class TestZipLuhnIntegration:
    """Integration tests for ZIP/Luhn example."""
    
    def test_zip_luhn_with_file_inputs(self, tmp_path, zip_luhn_mock_runtime):
        """Test ZIP/Luhn example using file inputs (simulating CLI usage)."""
        
        # Create spec file
        spec_file = tmp_path / "payment_spec.txt"
        spec_file.write_text("""
        Payment Processing Requirements
        
        The payment form must collect and validate user payment information:
        - Credit card numbers must pass Luhn algorithm validation
        - US ZIP codes are required for billing addresses
        - Clear error messages must be shown for invalid inputs
        """)
        
        # Create AC file
        ac_file = tmp_path / "acceptance_criteria.json"
        ac_data = {
            "acceptance_criteria": [
                "US ZIP code is required when saving a card",
                "Card number must pass Luhn; otherwise show a validation error"
            ]
        }
        ac_file.write_text(json.dumps(ac_data, indent=2))
        
        # Test file-based workflow
        from qa_generator.workflow import generate_qa_plan_from_files
        from qa_generator.models import Constraints
        
        constraints = Constraints(
            test_framework="playwright",
            priority_policy="risk_weighted"
        )
        
        test_plan = generate_qa_plan_from_files(
            project="payment-system",
            artifact_id="FILE-TEST-001",
            spec_file=spec_file,
            ac_file=ac_file,
            constraints=constraints,
            runtime=zip_luhn_mock_runtime,
            output_dir=tmp_path
        )
        
        # Validate test plan
        assert test_plan.project == "payment-system"
        assert test_plan.artifact_id == "FILE-TEST-001"
        assert len(test_plan.acceptance_criteria) == 2
        assert len(test_plan.scenarios) > 0
        assert len(test_plan.test_cases) > 0
        
        # Check that JSON file was created
        json_files = list(tmp_path.glob("qa_test_plan_*.json"))
        assert len(json_files) == 1
        
        # Check that test skeleton was created  
        skeleton_dirs = list(tmp_path.glob("test_skeletons_*"))
        assert len(skeleton_dirs) == 1
        
        skeleton_files = list(skeleton_dirs[0].glob("*.py"))
        assert len(skeleton_files) == 1