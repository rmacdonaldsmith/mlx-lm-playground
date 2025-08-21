"""
Data models and schemas for QA Test Scenario Generator.

These Pydantic models define the structure for inputs, outputs, and intermediate
artifacts in the 5-node workflow. All models include validation and JSON schema
generation for deterministic LLM interactions.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ==================== Input Models ====================

class APIInfo(BaseModel):
    """Optional API schema information for context (informational in v0)."""
    path: Optional[str] = Field(None, description="Path to OpenAPI/GraphQL schema file")
    text: Optional[str] = Field(None, description="Inline schema text")
    
    @model_validator(mode='after')
    def validate_api_source(self):
        if self.path and self.text:
            raise ValueError("Provide either 'path' or 'text', not both")
        return self


class Constraints(BaseModel):
    """Optional constraints for test generation."""
    test_framework: Optional[str] = Field(None, description="Target test framework (pytest, playwright, etc.)")
    environments: Optional[List[str]] = Field(None, description="Target environments (staging, prod, mobile)")
    priority_policy: Optional[Literal["risk_weighted", "uniform"]] = Field("risk_weighted", 
        description="Priority assignment policy")
    

class AcceptanceCriteria(BaseModel):
    """Normalized acceptance criteria with stable ID."""
    id: str = Field(..., description="Stable ID like AC1, AC2, etc.")
    text: str = Field(..., description="Original acceptance criteria text")
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v):
        if not v.startswith('AC') or not v[2:].isdigit():
            raise ValueError("ID must be in format 'AC{number}' (e.g., AC1, AC2)")
        return v


class RequirementsInput(BaseModel):
    """Complete input specification for QA generation."""
    project: str = Field(..., description="Project name for context")
    artifact_id: str = Field(..., description="Work item ID for traceability")
    spec_text: str = Field(..., description="PRD or user story text")
    acceptance_criteria: List[str] = Field(..., description="List of acceptance criteria strings")
    api: Optional[APIInfo] = None
    constraints: Optional[Constraints] = None
    
    @field_validator('acceptance_criteria')
    @classmethod
    def validate_non_empty_criteria(cls, v):
        if not v:
            raise ValueError("At least one acceptance criterion is required")
        return v


# ==================== Output Models ====================

class Scenario(BaseModel):
    """Test scenario definition."""
    id: str = Field(..., description="Unique scenario ID (SCN-001, SCN-002, etc.)")
    title: str = Field(..., description="Human-readable scenario title")
    type: Literal["functional", "integration", "e2e"] = Field(..., description="Scenario type")
    risk: Literal["low", "medium", "high"] = Field(..., description="Risk level")
    related_requirements: List[str] = Field(..., description="AC IDs this scenario covers")
    preconditions: Optional[str] = Field(None, description="Setup conditions")
    variants: Optional[List[str]] = Field(None, description="Variants (locale, auth state, etc.)")
    
    @field_validator('id')
    @classmethod
    def validate_scenario_id(cls, v):
        if not v.startswith('SCN-') or not v[4:].isdigit():
            raise ValueError("Scenario ID must be in format 'SCN-{number}' (e.g., SCN-001)")
        return v
    
    @field_validator('related_requirements')
    @classmethod
    def validate_related_requirements(cls, v):
        for req_id in v:
            if not req_id.startswith('AC') or not req_id[2:].isdigit():
                raise ValueError(f"Invalid AC reference: {req_id}")
        return v


class TestCase(BaseModel):
    """Concrete test case with explicit steps."""
    id: str = Field(..., description="Unique test case ID (TC-001, TC-002, etc.)")
    scenario_id: str = Field(..., description="Parent scenario ID")
    case_type: Literal["functional", "integration", "e2e", "negative", "boundary"] = Field(..., 
        description="Test case type")
    priority: Literal["P0", "P1", "P2", "P3"] = Field(..., description="Test priority")
    steps: List[str] = Field(..., description="Explicit test steps")
    data: Optional[Dict[str, Any]] = Field(None, description="Input data for test")
    expected: List[str] = Field(..., description="Expected results/assertions")
    negative: bool = Field(..., description="True if this is a negative test case")
    tags: Optional[List[str]] = Field(None, description="Additional tags for categorization")
    
    @field_validator('id')
    @classmethod
    def validate_test_case_id(cls, v):
        if not v.startswith('TC-') or not v[3:].isdigit():
            raise ValueError("Test case ID must be in format 'TC-{number}' (e.g., TC-001)")
        return v
    
    @field_validator('scenario_id')
    @classmethod
    def validate_scenario_ref(cls, v):
        if not v.startswith('SCN-') or not v[4:].isdigit():
            raise ValueError("scenario_id must reference valid scenario ID")
        return v
    
    @field_validator('steps')
    @classmethod
    def validate_non_empty_steps(cls, v):
        if not v:
            raise ValueError("Test case must have at least one step")
        return v
    
    @field_validator('expected')
    @classmethod
    def validate_non_empty_expected(cls, v):
        if not v:
            raise ValueError("Test case must have at least one expected result")
        return v


class OpenQuestion(BaseModel):
    """Questions about ambiguous or missing requirements."""
    id: str = Field(..., description="Question ID (Q-001, Q-002, etc.)")
    text: str = Field(..., description="The open question")
    blocking: bool = Field(..., description="True if this blocks test execution")
    related_requirements: Optional[List[str]] = Field(None, description="Related AC IDs")
    
    @field_validator('id')
    @classmethod
    def validate_question_id(cls, v):
        if not v.startswith('Q-') or not v[2:].isdigit():
            raise ValueError("Question ID must be in format 'Q-{number}' (e.g., Q-001)")
        return v


class CoverageMap(BaseModel):
    """Mapping from acceptance criteria to covering scenarios."""
    ac_to_scenarios: Dict[str, List[str]] = Field(..., 
        description="Map from AC ID to list of scenario IDs that cover it")
    
    def get_uncovered_acs(self, all_ac_ids: List[str]) -> List[str]:
        """Return list of AC IDs that have no covering scenarios."""
        return [ac_id for ac_id in all_ac_ids if ac_id not in self.ac_to_scenarios]
    
    def get_scenarios_for_ac(self, ac_id: str) -> List[str]:
        """Get all scenario IDs that cover a specific AC."""
        return self.ac_to_scenarios.get(ac_id, [])


class TestPlan(BaseModel):
    """Complete generated test plan artifact."""
    project: str = Field(..., description="Project name")
    artifact_id: str = Field(..., description="Work item ID")
    acceptance_criteria: List[AcceptanceCriteria] = Field(..., description="Normalized ACs")
    scenarios: List[Scenario] = Field(..., description="Generated test scenarios")
    test_cases: List[TestCase] = Field(..., description="Generated test cases")
    coverage_map: CoverageMap = Field(..., description="AC to scenario mapping")
    open_questions: List[OpenQuestion] = Field(..., description="Identified ambiguities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")


# ==================== Intermediate Models ====================

class ParsedRequirements(BaseModel):
    """Output from ParseRequirements node."""
    acceptance_criteria: List[AcceptanceCriteria]
    entities: Optional[Dict[str, Any]] = Field(None, description="Extracted entities/fields")
    constraints: Optional[Constraints] = None


class ScenarioGenerationResponse(BaseModel):
    """Expected JSON schema for LLM scenario generation."""
    scenarios: List[Scenario] = Field(..., description="Generated test scenarios")


class TestCaseGenerationResponse(BaseModel):
    """Expected JSON schema for LLM test case generation."""
    test_cases: List[TestCase] = Field(..., description="Generated test cases")


# ==================== Validation Helpers ====================

def validate_unique_ids(items: List[BaseModel], id_field: str = "id") -> None:
    """Validate that all items have unique IDs."""
    ids = [getattr(item, id_field) for item in items]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    if duplicates:
        raise ValueError(f"Duplicate IDs found: {duplicates}")


def validate_references(test_cases: List[TestCase], scenarios: List[Scenario]) -> None:
    """Validate that all test case scenario_id references are valid."""
    scenario_ids = {s.id for s in scenarios}
    invalid_refs = [tc.id for tc in test_cases if tc.scenario_id not in scenario_ids]
    if invalid_refs:
        raise ValueError(f"Test cases with invalid scenario references: {invalid_refs}")


# ==================== JSON Schema Generation ====================

def get_scenario_json_schema() -> Dict[str, Any]:
    """Get JSON schema for scenario generation prompt."""
    return ScenarioGenerationResponse.schema()


def get_test_case_json_schema() -> Dict[str, Any]:
    """Get JSON schema for test case generation prompt."""  
    return TestCaseGenerationResponse.schema()