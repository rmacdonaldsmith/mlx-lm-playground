"""
CoverageCritic Node (Deterministic)

Validates generated artifacts against G1 quality gates and builds coverage maps.
This is a deterministic node that performs strict validation and generates
open questions for ambiguities or gaps.
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple
import logging
from collections import defaultdict

from ..models import (
    AcceptanceCriteria,
    Scenario, 
    TestCase,
    CoverageMap,
    OpenQuestion
)
from ..exceptions import G1ValidationError

logger = logging.getLogger(__name__)


class G1Validator:
    """Validates artifacts against the G1 quality gates."""
    
    @staticmethod
    def validate_all_gates(
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario],
        test_cases: List[TestCase]
    ) -> None:
        """
        Validate all G1 gates. Raises G1ValidationError if any gate fails.
        
        G1 Gates:
        - G1.1: Every AC appears in coverage map with ≥1 scenario
        - G1.2: Every scenario has ≥1 test case  
        - G1.3: For each AC, there exists ≥1 positive and ≥1 negative test case
        - G1.4: All IDs are unique; no dangling references
        - G1.5: All artifacts are valid (handled by Pydantic models)
        """
        
        violations = []
        offending_ids = []
        
        # G1.1: AC coverage validation
        try:
            G1Validator.validate_g1_1(acceptance_criteria, scenarios)
        except G1ValidationError as e:
            violations.extend(e.violated_rules)
            offending_ids.extend(e.offending_ids)
        
        # G1.2: Scenario coverage validation  
        try:
            G1Validator.validate_g1_2(scenarios, test_cases)
        except G1ValidationError as e:
            violations.extend(e.violated_rules)
            offending_ids.extend(e.offending_ids)
        
        # G1.3: Positive/negative coverage validation
        try:
            G1Validator.validate_g1_3(acceptance_criteria, scenarios, test_cases)
        except G1ValidationError as e:
            violations.extend(e.violated_rules)
            offending_ids.extend(e.offending_ids)
        
        # G1.4: ID uniqueness and reference validation
        try:
            G1Validator.validate_g1_4(acceptance_criteria, scenarios, test_cases)
        except G1ValidationError as e:
            violations.extend(e.violated_rules)
            offending_ids.extend(e.offending_ids)
        
        # If any violations found, raise comprehensive error
        if violations:
            details = f"Found {len(violations)} G1 violations across {len(set(offending_ids))} items"
            raise G1ValidationError(violations, offending_ids, details)
        
        logger.info("All G1 quality gates passed successfully")
    
    @staticmethod
    def validate_g1_1(
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario]
    ) -> None:
        """G1.1: Every AC appears in coverage map with ≥1 scenario."""
        
        # Build coverage map
        coverage_map = CoverageCritic._build_coverage_map(scenarios)
        
        # Find uncovered ACs
        all_ac_ids = {ac.id for ac in acceptance_criteria}
        covered_ac_ids = set(coverage_map.ac_to_scenarios.keys())
        uncovered_acs = all_ac_ids - covered_ac_ids
        
        if uncovered_acs:
            raise G1ValidationError(
                violated_rules=["G1.1"],
                offending_ids=list(uncovered_acs),
                details=f"ACs without scenario coverage: {', '.join(sorted(uncovered_acs))}"
            )
    
    @staticmethod
    def validate_g1_2(scenarios: List[Scenario], test_cases: List[TestCase]) -> None:
        """G1.2: Every scenario has ≥1 test case."""
        
        scenario_ids = {s.id for s in scenarios}
        covered_scenario_ids = {tc.scenario_id for tc in test_cases}
        uncovered_scenarios = scenario_ids - covered_scenario_ids
        
        if uncovered_scenarios:
            raise G1ValidationError(
                violated_rules=["G1.2"],
                offending_ids=list(uncovered_scenarios),
                details=f"Scenarios without test cases: {', '.join(sorted(uncovered_scenarios))}"
            )
    
    @staticmethod
    def validate_g1_3(
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario],
        test_cases: List[TestCase]
    ) -> None:
        """G1.3: For each AC, there exists ≥1 positive and ≥1 negative test case."""
        
        # Build AC to test cases mapping
        ac_to_test_cases = G1Validator._build_ac_to_test_cases_map(scenarios, test_cases)
        
        violations = []
        offending_ids = []
        
        for ac in acceptance_criteria:
            ac_test_cases = ac_to_test_cases.get(ac.id, [])
            
            # Check for positive test cases
            positive_cases = [tc for tc in ac_test_cases if not tc.negative]
            if not positive_cases:
                violations.append("G1.3")
                offending_ids.append(ac.id)
                logger.warning(f"AC {ac.id} has no positive test cases")
            
            # Check for negative test cases
            negative_cases = [tc for tc in ac_test_cases if tc.negative]
            if not negative_cases:
                violations.append("G1.3")
                offending_ids.append(ac.id)
                logger.warning(f"AC {ac.id} has no negative test cases")
        
        if violations:
            unique_violations = list(set(violations))
            unique_offending = list(set(offending_ids))
            raise G1ValidationError(
                violated_rules=unique_violations,
                offending_ids=unique_offending,
                details=f"ACs missing positive/negative coverage: {', '.join(sorted(unique_offending))}"
            )
    
    @staticmethod
    def validate_g1_4(
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario],
        test_cases: List[TestCase]
    ) -> None:
        """G1.4: All IDs are unique; no dangling references."""
        
        violations = []
        offending_ids = []
        
        # Check AC ID uniqueness
        ac_ids = [ac.id for ac in acceptance_criteria]
        duplicate_ac_ids = G1Validator._find_duplicates(ac_ids)
        if duplicate_ac_ids:
            violations.append("G1.4")
            offending_ids.extend(duplicate_ac_ids)
        
        # Check scenario ID uniqueness
        scenario_ids = [s.id for s in scenarios]
        duplicate_scenario_ids = G1Validator._find_duplicates(scenario_ids)
        if duplicate_scenario_ids:
            violations.append("G1.4")
            offending_ids.extend(duplicate_scenario_ids)
        
        # Check test case ID uniqueness
        test_case_ids = [tc.id for tc in test_cases]
        duplicate_tc_ids = G1Validator._find_duplicates(test_case_ids)
        if duplicate_tc_ids:
            violations.append("G1.4")
            offending_ids.extend(duplicate_tc_ids)
        
        # Check scenario -> AC references
        valid_ac_ids = set(ac_ids)
        for scenario in scenarios:
            invalid_ac_refs = [ac_id for ac_id in scenario.related_requirements 
                             if ac_id not in valid_ac_ids]
            if invalid_ac_refs:
                violations.append("G1.4")
                offending_ids.append(scenario.id)
        
        # Check test case -> scenario references
        valid_scenario_ids = set(scenario_ids)
        for test_case in test_cases:
            if test_case.scenario_id not in valid_scenario_ids:
                violations.append("G1.4")
                offending_ids.append(test_case.id)
        
        if violations:
            unique_violations = list(set(violations))
            unique_offending = list(set(offending_ids))
            raise G1ValidationError(
                violated_rules=unique_violations,
                offending_ids=unique_offending,
                details=f"ID uniqueness or reference violations: {', '.join(sorted(unique_offending))}"
            )
    
    @staticmethod
    def _build_ac_to_test_cases_map(
        scenarios: List[Scenario], 
        test_cases: List[TestCase]
    ) -> Dict[str, List[TestCase]]:
        """Build mapping from AC ID to all test cases that cover it."""
        
        # First build scenario to test cases map
        scenario_to_cases = defaultdict(list)
        for tc in test_cases:
            scenario_to_cases[tc.scenario_id].append(tc)
        
        # Then build AC to test cases map via scenarios
        ac_to_cases = defaultdict(list)
        for scenario in scenarios:
            scenario_cases = scenario_to_cases[scenario.id]
            for ac_id in scenario.related_requirements:
                ac_to_cases[ac_id].extend(scenario_cases)
        
        return dict(ac_to_cases)
    
    @staticmethod
    def _find_duplicates(items: List[str]) -> List[str]:
        """Find duplicate items in a list."""
        seen = set()
        duplicates = set()
        
        for item in items:
            if item in seen:
                duplicates.add(item)
            seen.add(item)
        
        return list(duplicates)


class CoverageCritic:
    """
    Node 4: Coverage analysis and validation.
    
    Builds coverage maps, validates G1 gates, and generates open questions
    for ambiguities or missing specifications.
    """
    
    def __init__(self, runtime: Optional = None):
        """
        Args:
            runtime: Optional LLM runtime for generating open questions  
        """
        self.runtime = runtime
    
    def process(
        self,
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario],
        test_cases: List[TestCase]
    ) -> Tuple[CoverageMap, List[OpenQuestion]]:
        """
        Analyze coverage and validate quality gates.
        
        Args:
            acceptance_criteria: Normalized ACs
            scenarios: Generated scenarios
            test_cases: Generated test cases
            
        Returns:
            Tuple of (coverage_map, open_questions)
            
        Raises:
            G1ValidationError: If any quality gate fails
        """
        
        # Step 1: Validate G1 gates (fail fast if violated)
        G1Validator.validate_all_gates(acceptance_criteria, scenarios, test_cases)
        
        # Step 2: Build coverage map
        coverage_map = self._build_coverage_map(scenarios)
        
        # Step 3: Generate open questions
        open_questions = self._generate_open_questions(
            acceptance_criteria, scenarios, test_cases, coverage_map
        )
        
        logger.info(f"Coverage analysis complete: {len(coverage_map.ac_to_scenarios)} ACs covered, "
                   f"{len(open_questions)} open questions identified")
        
        return coverage_map, open_questions
    
    @staticmethod
    def _build_coverage_map(scenarios: List[Scenario]) -> CoverageMap:
        """Build coverage map from scenarios to ACs."""
        
        ac_to_scenarios = defaultdict(list)
        
        for scenario in scenarios:
            for ac_id in scenario.related_requirements:
                ac_to_scenarios[ac_id].append(scenario.id)
        
        # Sort scenario lists for deterministic output
        for ac_id in ac_to_scenarios:
            ac_to_scenarios[ac_id].sort()
        
        return CoverageMap(ac_to_scenarios=dict(ac_to_scenarios))
    
    def _generate_open_questions(
        self,
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario],
        test_cases: List[TestCase],
        coverage_map: CoverageMap
    ) -> List[OpenQuestion]:
        """Generate open questions about ambiguities or gaps."""
        
        questions = []
        
        # Analyze potential gaps and ambiguities
        questions.extend(self._identify_specification_gaps(acceptance_criteria))
        questions.extend(self._identify_error_handling_gaps(acceptance_criteria, test_cases))
        questions.extend(self._identify_integration_questions(acceptance_criteria, scenarios))
        questions.extend(self._identify_data_validation_questions(acceptance_criteria, test_cases))
        questions.extend(self._identify_edge_case_questions(scenarios, test_cases))
        
        # Add stable IDs to questions
        for i, question in enumerate(questions, 1):
            question.id = f"Q-{i:03d}"
        
        return questions
    
    def _identify_specification_gaps(
        self, 
        acceptance_criteria: List[AcceptanceCriteria]
    ) -> List[OpenQuestion]:
        """Identify gaps in specification clarity."""
        
        questions = []
        
        for ac in acceptance_criteria:
            text_lower = ac.text.lower()
            
            # Check for vague language
            if any(word in text_lower for word in ['appropriate', 'proper', 'suitable', 'reasonable']):
                questions.append(OpenQuestion(
                    id="",  # Will be set later
                    text=f"AC {ac.id} uses vague language. What specific behavior is expected?",
                    blocking=False,
                    related_requirements=[ac.id]
                ))
            
            # Check for missing error specifications
            if 'error' in text_lower and not any(word in text_lower for word in ['message', 'code', 'status']):
                questions.append(OpenQuestion(
                    id="",
                    text=f"AC {ac.id} mentions errors but doesn't specify error messages or codes. "
                         "What should users see when this error occurs?",
                    blocking=True,
                    related_requirements=[ac.id]
                ))
            
            # Check for validation without format specification
            if any(word in text_lower for word in ['validate', 'format', 'valid']) and 'format' not in text_lower:
                questions.append(OpenQuestion(
                    id="",
                    text=f"AC {ac.id} requires validation but doesn't specify the expected format. "
                         "What are the exact validation rules?",
                    blocking=True,
                    related_requirements=[ac.id]
                ))
        
        return questions
    
    def _identify_error_handling_gaps(
        self,
        acceptance_criteria: List[AcceptanceCriteria],
        test_cases: List[TestCase]
    ) -> List[OpenQuestion]:
        """Identify missing error handling specifications."""
        
        questions = []
        
        # Check for ACs that should have error cases but don't have negative tests
        for ac in acceptance_criteria:
            text_lower = ac.text.lower()
            
            # ACs that suggest validation should have error cases
            suggests_validation = any(word in text_lower for word in [
                'required', 'must', 'validate', 'check', 'verify', 'format'
            ])
            
            if suggests_validation:
                # Find test cases for this AC
                ac_test_cases = [tc for tc in test_cases 
                               if any(s for s in [tc.scenario_id] 
                                     if ac.id in getattr(tc, 'related_requirements', []))]
                
                negative_cases = [tc for tc in ac_test_cases if tc.negative]
                
                if not negative_cases:
                    questions.append(OpenQuestion(
                        id="",
                        text=f"AC {ac.id} suggests validation but has no negative test cases. "
                             "What should happen when validation fails?",
                        blocking=True,
                        related_requirements=[ac.id]
                    ))
        
        return questions
    
    def _identify_integration_questions(
        self,
        acceptance_criteria: List[AcceptanceCriteria],
        scenarios: List[Scenario]
    ) -> List[OpenQuestion]:
        """Identify questions about system integrations."""
        
        questions = []
        
        # Look for integration scenarios without clear specifications
        integration_scenarios = [s for s in scenarios if s.type == "integration"]
        
        for scenario in integration_scenarios:
            if any(word in scenario.title.lower() for word in ['api', 'service', 'external', 'third-party']):
                questions.append(OpenQuestion(
                    id="",
                    text=f"Scenario {scenario.id} involves external integration. "
                         "What should happen if the external service is unavailable?",
                    blocking=False,
                    related_requirements=scenario.related_requirements
                ))
        
        return questions
    
    def _identify_data_validation_questions(
        self,
        acceptance_criteria: List[AcceptanceCriteria],
        test_cases: List[TestCase]
    ) -> List[OpenQuestion]:
        """Identify questions about data validation edge cases."""
        
        questions = []
        
        # Look for data validation patterns
        for ac in acceptance_criteria:
            text_lower = ac.text.lower()
            
            # ZIP code validation
            if 'zip' in text_lower and 'code' in text_lower:
                questions.append(OpenQuestion(
                    id="",
                    text=f"AC {ac.id} mentions ZIP codes. Should the system accept "
                         "international postal codes, or only US ZIP codes?",
                    blocking=False,
                    related_requirements=[ac.id]
                ))
            
            # Email validation
            if 'email' in text_lower:
                questions.append(OpenQuestion(
                    id="",
                    text=f"AC {ac.id} involves email validation. What specific email "
                         "format rules should be applied (RFC compliance, length limits, etc.)?",
                    blocking=False,
                    related_requirements=[ac.id]
                ))
            
            # Phone number validation
            if 'phone' in text_lower:
                questions.append(OpenQuestion(
                    id="",
                    text=f"AC {ac.id} mentions phone numbers. Should the system accept "
                         "international formats, extensions, or only domestic numbers?",
                    blocking=False,
                    related_requirements=[ac.id]
                ))
        
        return questions
    
    def _identify_edge_case_questions(
        self,
        scenarios: List[Scenario],
        test_cases: List[TestCase]
    ) -> List[OpenQuestion]:
        """Identify questions about potential edge cases."""
        
        questions = []
        
        # Look for boundary scenarios that might need clarification
        boundary_cases = [tc for tc in test_cases if tc.case_type == "boundary"]
        
        if len(boundary_cases) < len(scenarios) * 0.2:  # Less than 20% boundary coverage
            questions.append(OpenQuestion(
                id="",
                text="Limited boundary test coverage detected. Should additional edge cases "
                     "be considered (empty inputs, maximum lengths, special characters)?",
                blocking=False,
                related_requirements=[]
            ))
        
        # Check for concurrent access scenarios
        has_concurrent_tests = any(
            'concurrent' in tc.steps[0].lower() if tc.steps else False
            for tc in test_cases
        )
        
        if not has_concurrent_tests and len(test_cases) > 10:
            questions.append(OpenQuestion(
                id="",
                text="No concurrent access scenarios detected. Should the system "
                     "handle multiple simultaneous users or operations?",
                blocking=False,
                related_requirements=[]
            ))
        
        return questions


# Convenience functions

def validate_coverage(
    acceptance_criteria: List[AcceptanceCriteria],
    scenarios: List[Scenario],
    test_cases: List[TestCase]
) -> Tuple[CoverageMap, List[OpenQuestion]]:
    """Convenience function for coverage validation."""
    critic = CoverageCritic()
    return critic.process(acceptance_criteria, scenarios, test_cases)


def check_g1_compliance(
    acceptance_criteria: List[AcceptanceCriteria],
    scenarios: List[Scenario],
    test_cases: List[TestCase]
) -> bool:
    """Check if artifacts comply with G1 gates without raising exceptions."""
    try:
        G1Validator.validate_all_gates(acceptance_criteria, scenarios, test_cases)
        return True
    except G1ValidationError:
        return False