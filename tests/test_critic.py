"""Test CoverageCritic node and G1 validation."""

import pytest

from qa_generator.nodes.critic import CoverageCritic, G1Validator
from qa_generator.models import AcceptanceCriteria, Scenario, TestCase, CoverageMap
from qa_generator.exceptions import G1ValidationError


class TestG1Validator:
    """Test the G1 quality gate validator."""
    
    def test_g1_1_validation_success(self, sample_normalized_acs):
        """Test G1.1 passes when all ACs have scenario coverage."""
        scenarios = [
            Scenario(id="SCN-001", title="Test AC1", type="functional", risk="low", 
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Test AC2", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        # Should not raise exception
        G1Validator.validate_g1_1(sample_normalized_acs, scenarios)
    
    def test_g1_1_validation_failure(self, sample_normalized_acs):
        """Test G1.1 fails when some ACs lack scenario coverage.""" 
        scenarios = [
            # Only covers AC1, AC2 is missing
            Scenario(id="SCN-001", title="Test AC1", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None)
        ]
        
        with pytest.raises(G1ValidationError) as exc_info:
            G1Validator.validate_g1_1(sample_normalized_acs, scenarios)
        
        error = exc_info.value
        assert "G1.1" in error.violated_rules
        assert "AC2" in error.offending_ids
        assert "without scenario coverage" in error.details
    
    def test_g1_2_validation_success(self):
        """Test G1.2 passes when all scenarios have test cases."""
        scenarios = [
            Scenario(id="SCN-001", title="Test", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Test", type="functional", risk="low", 
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-002", scenario_id="SCN-002", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None)
        ]
        
        # Should not raise exception
        G1Validator.validate_g1_2(scenarios, test_cases)
    
    def test_g1_2_validation_failure(self):
        """Test G1.2 fails when scenarios lack test cases."""
        scenarios = [
            Scenario(id="SCN-001", title="Test", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Test", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            # Only covers SCN-001, SCN-002 is missing
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None)
        ]
        
        with pytest.raises(G1ValidationError) as exc_info:
            G1Validator.validate_g1_2(scenarios, test_cases)
        
        error = exc_info.value
        assert "G1.2" in error.violated_rules
        assert "SCN-002" in error.offending_ids
    
    def test_g1_3_validation_success(self, sample_normalized_acs):
        """Test G1.3 passes when ACs have both positive and negative test cases."""
        scenarios = [
            Scenario(id="SCN-001", title="Positive AC1", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Negative AC1", type="functional", risk="low", 
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-003", title="Positive AC2", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None),
            Scenario(id="SCN-004", title="Negative AC2", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            # AC1 positive
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            # AC1 negative  
            TestCase(id="TC-002", scenario_id="SCN-002", case_type="negative", priority="P2",
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None),
            # AC2 positive
            TestCase(id="TC-003", scenario_id="SCN-003", case_type="functional", priority="P1", 
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            # AC2 negative
            TestCase(id="TC-004", scenario_id="SCN-004", case_type="negative", priority="P2",
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None)
        ]
        
        # Should not raise exception
        G1Validator.validate_g1_3(sample_normalized_acs, scenarios, test_cases)
    
    def test_g1_3_validation_failure_missing_negative(self, sample_normalized_acs):
        """Test G1.3 fails when ACs lack negative test cases."""
        scenarios = [
            Scenario(id="SCN-001", title="Only positive", type="functional", risk="low",
                    related_requirements=["AC1", "AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            # Only positive test cases, no negative
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None)
        ]
        
        with pytest.raises(G1ValidationError) as exc_info:
            G1Validator.validate_g1_3(sample_normalized_acs, scenarios, test_cases)
        
        error = exc_info.value
        assert "G1.3" in error.violated_rules
        assert "AC1" in error.offending_ids
        assert "AC2" in error.offending_ids
        assert "missing positive/negative coverage" in error.details
    
    def test_g1_4_validation_success(self, sample_normalized_acs):
        """Test G1.4 passes with unique IDs and valid references."""
        scenarios = [
            Scenario(id="SCN-001", title="Test", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Test", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-002", scenario_id="SCN-002", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None)
        ]
        
        # Should not raise exception
        G1Validator.validate_g1_4(sample_normalized_acs, scenarios, test_cases)
    
    def test_g1_4_validation_failure_duplicate_ids(self, sample_normalized_acs):
        """Test G1.4 fails with duplicate IDs."""
        scenarios = [
            Scenario(id="SCN-001", title="First", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-001", title="Duplicate ID", type="functional", risk="low", 
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = []
        
        with pytest.raises(G1ValidationError) as exc_info:
            G1Validator.validate_g1_4(sample_normalized_acs, scenarios, test_cases)
        
        error = exc_info.value
        assert "G1.4" in error.violated_rules
        assert "SCN-001" in error.offending_ids
    
    def test_g1_4_validation_failure_dangling_references(self, sample_normalized_acs):
        """Test G1.4 fails with dangling references."""
        scenarios = [
            # References non-existent AC3
            Scenario(id="SCN-001", title="Test", type="functional", risk="low",
                    related_requirements=["AC3"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            # References non-existent scenario  
            TestCase(id="TC-001", scenario_id="SCN-999", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None)
        ]
        
        with pytest.raises(G1ValidationError) as exc_info:
            G1Validator.validate_g1_4(sample_normalized_acs, scenarios, test_cases)
        
        error = exc_info.value
        assert "G1.4" in error.violated_rules
        # Should contain both SCN-001 (invalid AC ref) and TC-001 (invalid scenario ref)
        assert "SCN-001" in error.offending_ids
        assert "TC-001" in error.offending_ids
    
    def test_validate_all_gates_success(self, sample_normalized_acs):
        """Test that all G1 gates pass with valid artifacts."""
        scenarios = [
            Scenario(id="SCN-001", title="AC1 positive", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="AC1 negative", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-003", title="AC2 positive", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None),
            Scenario(id="SCN-004", title="AC2 negative", type="functional", risk="low", 
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-002", scenario_id="SCN-002", case_type="negative", priority="P2", 
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None),
            TestCase(id="TC-003", scenario_id="SCN-003", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-004", scenario_id="SCN-004", case_type="negative", priority="P2",
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None)
        ]
        
        # Should not raise exception
        G1Validator.validate_all_gates(sample_normalized_acs, scenarios, test_cases)


class TestCoverageCritic:
    """Test the CoverageCritic node."""
    
    def test_build_coverage_map(self):
        """Test coverage map building."""
        scenarios = [
            Scenario(id="SCN-001", title="Test", type="functional", risk="low", 
                    related_requirements=["AC1", "AC2"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="Test", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        coverage_map = CoverageCritic._build_coverage_map(scenarios)
        
        assert coverage_map.ac_to_scenarios["AC1"] == ["SCN-001"]
        assert set(coverage_map.ac_to_scenarios["AC2"]) == {"SCN-001", "SCN-002"}
    
    def test_process_with_valid_artifacts(self, sample_normalized_acs):
        """Test successful processing with valid artifacts."""
        scenarios = [
            Scenario(id="SCN-001", title="AC1 positive", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-002", title="AC1 negative", type="functional", risk="low",
                    related_requirements=["AC1"], preconditions=None, variants=None),
            Scenario(id="SCN-003", title="AC2 positive", type="functional", risk="low", 
                    related_requirements=["AC2"], preconditions=None, variants=None),
            Scenario(id="SCN-004", title="AC2 negative", type="functional", risk="low",
                    related_requirements=["AC2"], preconditions=None, variants=None)
        ]
        
        test_cases = [
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-002", scenario_id="SCN-002", case_type="negative", priority="P2",
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None),
            TestCase(id="TC-003", scenario_id="SCN-003", case_type="functional", priority="P1",
                    steps=["Step"], expected=["Result"], negative=False, data=None, tags=None),
            TestCase(id="TC-004", scenario_id="SCN-004", case_type="negative", priority="P2",
                    steps=["Step"], expected=["Error"], negative=True, data=None, tags=None)
        ]
        
        critic = CoverageCritic()
        coverage_map, open_questions = critic.process(sample_normalized_acs, scenarios, test_cases)
        
        # Validate coverage map
        assert len(coverage_map.ac_to_scenarios) == 2
        assert "AC1" in coverage_map.ac_to_scenarios
        assert "AC2" in coverage_map.ac_to_scenarios
        
        # Validate open questions generated
        assert isinstance(open_questions, list)
        # Should have some questions about ambiguities
        question_texts = [q.text for q in open_questions]
        
        # Check for specific question types based on AC content
        zip_questions = [q for q in open_questions if "ZIP" in q.text or "postal" in q.text]
        luhn_questions = [q for q in open_questions if "Luhn" in q.text or "card" in q.text]
        
        # Should have questions about ZIP code formats and card validation
        assert len(zip_questions) > 0 or len(luhn_questions) > 0
    
    def test_generate_open_questions_specification_gaps(self):
        """Test generation of specification gap questions."""
        # ACs with vague language
        acs = [
            AcceptanceCriteria(id="AC1", text="System should show appropriate error message"),
            AcceptanceCriteria(id="AC2", text="Validation must be proper and reasonable")
        ]
        
        critic = CoverageCritic()
        questions = critic._identify_specification_gaps(acs)
        
        # Should identify vague language
        vague_questions = [q for q in questions if "vague language" in q.text]
        assert len(vague_questions) == 2
        
        # Check question structure
        for question in vague_questions:
            assert question.text
            assert not question.blocking  # Vague language is non-blocking
            assert len(question.related_requirements) == 1
    
    def test_generate_open_questions_error_handling(self):
        """Test generation of error handling questions.""" 
        acs = [
            AcceptanceCriteria(id="AC1", text="System must validate email format"),
            AcceptanceCriteria(id="AC2", text="Required fields should show error when empty")
        ]
        
        # Test cases without negative coverage
        test_cases = [
            TestCase(id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                    steps=["Enter email"], expected=["Success"], negative=False, data=None, tags=None)
            # Missing negative test case
        ]
        
        critic = CoverageCritic()
        questions = critic._identify_error_handling_gaps(acs, test_cases)
        
        # Should identify missing error handling
        error_questions = [q for q in questions if "validation fails" in q.text]
        assert len(error_questions) >= 1
        
        # Error handling gaps should be blocking
        for question in error_questions:
            assert question.blocking