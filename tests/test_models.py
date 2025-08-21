"""Test data models and validation."""

import pytest
from pydantic import ValidationError

from qa_generator.models import (
    RequirementsInput,
    AcceptanceCriteria,
    Scenario,
    TestCase,
    CoverageMap,
    OpenQuestion,
    TestPlan,
    validate_unique_ids,
    validate_references
)


class TestRequirementsInput:
    """Test RequirementsInput model validation."""
    
    def test_valid_requirements(self):
        """Test creating valid requirements input."""
        req = RequirementsInput(
            project="test-project",
            artifact_id="JIRA-123",
            spec_text="Test specification",
            acceptance_criteria=["AC requirement 1", "AC requirement 2"]
        )
        
        assert req.project == "test-project"
        assert req.artifact_id == "JIRA-123"
        assert len(req.acceptance_criteria) == 2
    
    def test_empty_acceptance_criteria_fails(self):
        """Test that empty acceptance criteria list fails validation."""
        with pytest.raises(ValidationError, match="At least one acceptance criterion is required"):
            RequirementsInput(
                project="test",
                artifact_id="TEST-1", 
                spec_text="Test",
                acceptance_criteria=[]  # Empty list should fail
            )


class TestAcceptanceCriteria:
    """Test AcceptanceCriteria model validation."""
    
    def test_valid_ac_id_format(self):
        """Test valid AC ID formats."""
        valid_ids = ["AC1", "AC2", "AC10", "AC999"]
        
        for ac_id in valid_ids:
            ac = AcceptanceCriteria(id=ac_id, text="Test criterion")
            assert ac.id == ac_id
    
    def test_invalid_ac_id_formats(self):
        """Test invalid AC ID formats."""
        invalid_ids = ["A1", "AC", "1AC", "AC-1", "ac1"]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValidationError, match="ID must be in format 'AC\\{number\\}'"):
                AcceptanceCriteria(id=invalid_id, text="Test")


class TestScenario:
    """Test Scenario model validation."""
    
    def test_valid_scenario(self):
        """Test creating valid scenario."""
        scenario = Scenario(
            id="SCN-001",
            title="Test scenario", 
            type="functional",
            risk="medium",
            related_requirements=["AC1", "AC2"],
            preconditions="Setup condition",
            variants=["mobile", "desktop"]
        )
        
        assert scenario.id == "SCN-001"
        assert scenario.type == "functional"
        assert scenario.risk == "medium"
        assert len(scenario.related_requirements) == 2
    
    def test_invalid_scenario_id(self):
        """Test invalid scenario ID format."""
        with pytest.raises(ValidationError, match="Scenario ID must be in format 'SCN-\\{number\\}'"):
            Scenario(
                id="SC-001",  # Invalid format
                title="Test",
                type="functional",
                risk="low",
                related_requirements=["AC1"]
            )
    
    def test_invalid_ac_references(self):
        """Test invalid AC reference format."""
        with pytest.raises(ValidationError, match="Invalid AC reference"):
            Scenario(
                id="SCN-001",
                title="Test",
                type="functional", 
                risk="low",
                related_requirements=["INVALID-REF"]  # Invalid format
            )


class TestTestCase:
    """Test TestCase model validation."""
    
    def test_valid_test_case(self):
        """Test creating valid test case."""
        tc = TestCase(
            id="TC-001",
            scenario_id="SCN-001",
            case_type="functional",
            priority="P1",
            steps=["Step 1", "Step 2"],
            data={"input": "value"},
            expected=["Expected 1", "Expected 2"],
            negative=False,
            tags=["tag1", "tag2"]
        )
        
        assert tc.id == "TC-001"
        assert tc.scenario_id == "SCN-001"
        assert tc.priority == "P1"
        assert len(tc.steps) == 2
        assert len(tc.expected) == 2
    
    def test_empty_steps_fails(self):
        """Test that empty steps list fails validation."""
        with pytest.raises(ValidationError, match="Test case must have at least one step"):
            TestCase(
                id="TC-001",
                scenario_id="SCN-001",
                case_type="functional",
                priority="P1",
                steps=[],  # Empty steps should fail
                expected=["Result"],
                negative=False
            )
    
    def test_empty_expected_fails(self):
        """Test that empty expected results fails validation."""
        with pytest.raises(ValidationError, match="Test case must have at least one expected result"):
            TestCase(
                id="TC-001",
                scenario_id="SCN-001", 
                case_type="functional",
                priority="P1",
                steps=["Step 1"],
                expected=[],  # Empty expected should fail
                negative=False
            )


class TestCoverageMap:
    """Test CoverageMap functionality."""
    
    def test_coverage_map_operations(self):
        """Test coverage map operations."""
        coverage = CoverageMap(ac_to_scenarios={
            "AC1": ["SCN-001", "SCN-002"],
            "AC2": ["SCN-003"]
        })
        
        # Test getting scenarios for AC
        assert coverage.get_scenarios_for_ac("AC1") == ["SCN-001", "SCN-002"]
        assert coverage.get_scenarios_for_ac("AC2") == ["SCN-003"]
        assert coverage.get_scenarios_for_ac("AC999") == []
        
        # Test finding uncovered ACs
        all_ac_ids = ["AC1", "AC2", "AC3"]
        uncovered = coverage.get_uncovered_acs(all_ac_ids)
        assert uncovered == ["AC3"]


class TestValidationHelpers:
    """Test validation helper functions."""
    
    def test_validate_unique_ids(self):
        """Test unique ID validation."""
        scenarios = [
            Scenario(id="SCN-001", title="Test 1", type="functional", risk="low", related_requirements=["AC1"]),
            Scenario(id="SCN-002", title="Test 2", type="functional", risk="low", related_requirements=["AC1"])
        ]
        
        # Should not raise exception
        validate_unique_ids(scenarios)
        
        # Add duplicate
        scenarios.append(Scenario(id="SCN-001", title="Duplicate", type="functional", risk="low", related_requirements=["AC1"]))
        
        with pytest.raises(ValueError, match="Duplicate IDs found"):
            validate_unique_ids(scenarios)
    
    def test_validate_references(self):
        """Test reference validation between test cases and scenarios."""
        scenarios = [
            Scenario(id="SCN-001", title="Test", type="functional", risk="low", related_requirements=["AC1"])
        ]
        
        valid_test_cases = [
            TestCase(
                id="TC-001", scenario_id="SCN-001", case_type="functional", priority="P1",
                steps=["Step"], expected=["Result"], negative=False
            )
        ]
        
        # Should not raise exception
        validate_references(valid_test_cases, scenarios)
        
        invalid_test_cases = [
            TestCase(
                id="TC-002", scenario_id="SCN-999", case_type="functional", priority="P1",  # Invalid reference
                steps=["Step"], expected=["Result"], negative=False
            )
        ]
        
        with pytest.raises(ValueError, match="invalid scenario references"):
            validate_references(invalid_test_cases, scenarios)