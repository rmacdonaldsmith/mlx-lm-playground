"""
CaseGenerator Node (LLM-based)

Expands test scenarios into concrete test cases with explicit steps,
data inputs, and expected results. Assigns priorities and test types.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
import logging

from ..runtime import LLMRuntime
from ..validation import generate_with_validation
from ..models import (
    Scenario,
    TestCase,
    TestCaseGenerationResponse,
    Constraints
)

logger = logging.getLogger(__name__)


class CaseGenerator:
    """
    Node 3: Generate concrete test cases from scenarios using LLM.
    
    Expands each scenario into one or more detailed test cases with:
    - Explicit step-by-step instructions
    - Input data specifications  
    - Expected results/assertions
    - Priority assignments based on risk and policy
    - Proper negative/positive classification
    """
    
    def __init__(self, runtime: LLMRuntime, constraints: Optional[Constraints] = None):
        self.runtime = runtime
        self.constraints = constraints or Constraints()
    
    def process(self, scenarios: List[Scenario]) -> List[TestCase]:
        """
        Generate concrete test cases from scenarios.
        
        Args:
            scenarios: List of test scenarios to expand
            
        Returns:
            List of generated test cases
        """
        
        all_test_cases = []
        
        # Process scenarios in batches to avoid context limits
        batch_size = 5
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            
            # Generate test cases for this batch
            batch_cases = self._generate_batch_test_cases(batch)
            
            # Post-process the batch
            processed_cases = self._post_process_test_cases(batch_cases, batch)
            
            all_test_cases.extend(processed_cases)
        
        logger.info(f"Generated {len(all_test_cases)} test cases from {len(scenarios)} scenarios")
        return all_test_cases
    
    def _generate_batch_test_cases(self, scenarios: List[Scenario]) -> List[TestCase]:
        """Generate test cases for a batch of scenarios."""
        
        prompt = self._build_test_case_prompt(scenarios)
        
        # Generate test cases using LLM with validation
        response = generate_with_validation(
            runtime=self.runtime,
            prompt=prompt,
            response_model=TestCaseGenerationResponse,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=4000   # More tokens needed for detailed test cases
        )
        
        return response.test_cases
    
    def _build_test_case_prompt(self, scenarios: List[Scenario]) -> str:
        """Build the LLM prompt for test case generation."""
        
        # Build scenarios description
        scenarios_text = []
        for scenario in scenarios:
            requirements = ", ".join(scenario.related_requirements)
            scenarios_text.append(
                f"- {scenario.id}: {scenario.title} (Type: {scenario.type}, Risk: {scenario.risk}, ACs: {requirements})"
            )
        
        scenarios_str = "\n".join(scenarios_text)
        
        # Get priority policy
        priority_policy = self.constraints.priority_policy or "risk_weighted"
        
        # Build framework-specific guidance
        framework_guidance = ""
        if self.constraints.test_framework:
            framework_guidance = f"\n\nTEST FRAMEWORK: {self.constraints.test_framework}\n" + \
                               self._get_framework_guidance(self.constraints.test_framework)
        
        prompt = f"""You are a QA test case generator. Create detailed, executable test cases for the following scenarios.

SCENARIOS TO EXPAND:
{scenarios_str}

REQUIREMENTS:
1. For each scenario, generate 1-3 concrete test cases
2. Each test case must have:
   - ID: TC-001, TC-002, TC-003, etc. (sequential numbering)
   - scenario_id: Reference to parent scenario
   - case_type: "functional", "integration", "e2e", "negative", or "boundary"
   - priority: "P0", "P1", "P2", or "P3"
   - steps: Detailed step-by-step instructions (explicit actions)
   - data: Input data needed (JSON object or null)
   - expected: List of expected results/assertions
   - negative: true if this tests error/failure cases, false for happy path
   - tags: Optional categorization tags

3. PRIORITY POLICY ({priority_policy}):
   """ + self._get_priority_guidance(priority_policy) + f"""

4. STEP REQUIREMENTS:
   - Be explicit and actionable ("Click the Submit button", not "Submit the form")
   - Include setup steps if needed
   - Specify exact data to enter
   - Include verification steps
   - Avoid flaky patterns (no hard-coded waits, use proper waits)

5. DATA REQUIREMENTS:
   - Provide realistic test data
   - Include both valid and invalid data for negative tests
   - Use JSON format for structured data
   - Include edge cases (empty strings, special characters, boundary values)

6. EXPECTED RESULTS:
   - Be specific and measurable
   - Include UI changes, messages, API responses
   - Specify error messages for negative tests
   - Include state changes (database, session, etc.)

7. NEGATIVE TEST CLASSIFICATION:
   - Set negative=true for error conditions, validation failures, unauthorized access
   - Set negative=false for happy path and valid boundary cases

{framework_guidance}

Generate comprehensive test cases that provide thorough coverage of each scenario.

Return ONLY valid JSON matching this exact structure (no markdown, no explanations):
{{
  "test_cases": [
    {{
      "id": "TC-001", 
      "scenario_id": "SCN-001",
      "title": "Test case title",
      "description": "Detailed test case description",
      "priority": "high",
      "steps": [
        {{"action": "Action description", "expected": "Expected result"}}
      ],
      "test_data": {{"key": "value"}},
      "environment": "staging",
      "tags": ["smoke", "regression"]
    }}
  ]
}}"""
        
        return prompt
    
    def _get_priority_guidance(self, policy: str) -> str:
        """Get priority assignment guidance based on policy."""
        
        if policy == "risk_weighted":
            return """
   - P0: Critical path, high-risk scenarios (auth, payments, data integrity, security)
   - P1: Important functionality, medium-risk scenarios
   - P2: Standard functionality, low-risk scenarios
   - P3: Edge cases, cosmetic issues, nice-to-have features
   
   Higher risk scenarios should generally get higher priority."""
        
        elif policy == "uniform":
            return """
   - P1: Default priority for most test cases
   - P2: Negative test cases and error scenarios
   - P3: Boundary cases and edge scenarios
   
   Distribute priorities evenly unless specified otherwise."""
        
        else:
            return """
   - Assign priorities based on business importance and risk
   - P0: Critical, P1: Important, P2: Standard, P3: Nice-to-have"""
    
    def _get_framework_guidance(self, framework: str) -> str:
        """Get framework-specific guidance for test case generation."""
        
        guidance = {
            "playwright": """
- Write steps compatible with Playwright (page.click(), page.fill(), expect())
- Use proper locators (data-testid, role-based selectors)
- Include async/await patterns where needed
- Add wait conditions for dynamic content""",
            
            "selenium": """
- Write steps compatible with Selenium WebDriver
- Use explicit waits (WebDriverWait, expected_conditions)
- Specify element locators (ID, CSS, XPath)
- Include browser setup/teardown considerations""",
            
            "pytest": """
- Structure steps for pytest test functions
- Include assertion statements using assert
- Consider fixture dependencies
- Use descriptive test function names""",
            
            "jest": """
- Write steps for Jest/React Testing Library
- Use proper queries (getByRole, getByTestId)
- Include user interaction events
- Structure for async testing patterns""",
            
            "cypress": """
- Write steps using Cypress commands (cy.visit(), cy.get(), cy.should())
- Use Cypress best practices for element selection
- Include proper assertions with .should()
- Consider page load and network handling"""
        }
        
        return guidance.get(framework.lower(), "Write framework-agnostic test steps")
    
    def _post_process_test_cases(
        self, 
        test_cases: List[TestCase],
        scenarios: List[Scenario]
    ) -> List[TestCase]:
        """Post-process generated test cases for quality and consistency."""
        
        # Step 1: Fix test case IDs
        test_cases = self._fix_test_case_ids(test_cases)
        
        # Step 2: Validate scenario references
        test_cases = self._validate_scenario_references(test_cases, scenarios)
        
        # Step 3: Ensure scenario coverage
        test_cases = self._ensure_scenario_coverage(test_cases, scenarios)
        
        # Step 4: Refine priority assignments
        test_cases = self._refine_priorities(test_cases, scenarios)
        
        # Step 5: Validate test case completeness
        test_cases = self._validate_test_case_completeness(test_cases)
        
        return test_cases
    
    def _fix_test_case_ids(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Ensure test case IDs are properly formatted and sequential."""
        
        fixed_cases = []
        for i, test_case in enumerate(test_cases, 1):
            test_id = f"TC-{i:03d}"
            
            # Create new test case with corrected ID
            fixed_case = TestCase(
                id=test_id,
                scenario_id=test_case.scenario_id,
                case_type=test_case.case_type,
                priority=test_case.priority,
                steps=test_case.steps,
                data=test_case.data,
                expected=test_case.expected,
                negative=test_case.negative,
                tags=test_case.tags
            )
            
            fixed_cases.append(fixed_case)
        
        return fixed_cases
    
    def _validate_scenario_references(
        self, 
        test_cases: List[TestCase],
        scenarios: List[Scenario]
    ) -> List[TestCase]:
        """Validate that test case scenario references are valid."""
        
        valid_scenario_ids = {scenario.id for scenario in scenarios}
        valid_cases = []
        
        for test_case in test_cases:
            if test_case.scenario_id in valid_scenario_ids:
                valid_cases.append(test_case)
            else:
                logger.warning(f"Test case {test_case.id} references invalid scenario {test_case.scenario_id}")
        
        return valid_cases
    
    def _ensure_scenario_coverage(
        self, 
        test_cases: List[TestCase],
        scenarios: List[Scenario]
    ) -> List[TestCase]:
        """Ensure every scenario has at least one test case."""
        
        # Find scenarios without test cases
        covered_scenarios = {tc.scenario_id for tc in test_cases}
        uncovered_scenarios = [s for s in scenarios if s.id not in covered_scenarios]
        
        # Create minimal test cases for uncovered scenarios
        additional_cases = []
        next_id = len(test_cases) + 1
        
        for scenario in uncovered_scenarios:
            minimal_case = TestCase(
                id=f"TC-{next_id:03d}",
                scenario_id=scenario.id,
                case_type=scenario.type,
                priority="P2",
                steps=[f"Execute test scenario: {scenario.title}"],
                data=None,
                expected=[f"Scenario {scenario.id} completes successfully"],
                negative=self._is_negative_scenario(scenario),
                tags=[f"auto-generated"]
            )
            
            additional_cases.append(minimal_case)
            next_id += 1
        
        if additional_cases:
            logger.info(f"Added {len(additional_cases)} minimal test cases for uncovered scenarios")
        
        return test_cases + additional_cases
    
    def _is_negative_scenario(self, scenario: Scenario) -> bool:
        """Determine if a scenario is likely a negative test case."""
        title_lower = scenario.title.lower()
        negative_indicators = [
            'invalid', 'error', 'fail', 'missing', 'empty', 'wrong',
            'unauthorized', 'forbidden', 'denied', 'timeout', 'limit'
        ]
        
        return any(indicator in title_lower for indicator in negative_indicators)
    
    def _refine_priorities(
        self, 
        test_cases: List[TestCase],
        scenarios: List[Scenario]
    ) -> List[TestCase]:
        """Refine priority assignments based on scenario risk and policy."""
        
        # Build scenario risk map
        scenario_risk = {s.id: s.risk for s in scenarios}
        
        policy = self.constraints.priority_policy or "risk_weighted"
        
        refined_cases = []
        for test_case in test_cases:
            scenario_risk_level = scenario_risk.get(test_case.scenario_id, "medium")
            
            # Calculate priority based on policy
            if policy == "risk_weighted":
                if scenario_risk_level == "high":
                    priority = "P0" if not test_case.negative else "P1"
                elif scenario_risk_level == "medium":
                    priority = "P1" if not test_case.negative else "P2"
                else:  # low risk
                    priority = "P2" if not test_case.negative else "P3"
            else:  # uniform policy
                if test_case.case_type == "boundary":
                    priority = "P3"
                elif test_case.negative:
                    priority = "P2"
                else:
                    priority = "P1"
            
            # Create refined test case
            refined_case = TestCase(
                id=test_case.id,
                scenario_id=test_case.scenario_id,
                case_type=test_case.case_type,
                priority=priority,
                steps=test_case.steps,
                data=test_case.data,
                expected=test_case.expected,
                negative=test_case.negative,
                tags=test_case.tags
            )
            
            refined_cases.append(refined_case)
        
        return refined_cases
    
    def _validate_test_case_completeness(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Validate that test cases have all required fields properly filled."""
        
        complete_cases = []
        
        for test_case in test_cases:
            # Ensure steps are not empty
            if not test_case.steps:
                logger.warning(f"Test case {test_case.id} has no steps, adding placeholder")
                test_case.steps = ["Execute test case"]
            
            # Ensure expected results are not empty
            if not test_case.expected:
                logger.warning(f"Test case {test_case.id} has no expected results, adding placeholder")
                test_case.expected = ["Test case completes successfully"]
            
            complete_cases.append(test_case)
        
        return complete_cases


# Convenience function

def generate_test_cases(
    scenarios: List[Scenario], 
    runtime: LLMRuntime,
    constraints: Optional[Constraints] = None
) -> List[TestCase]:
    """Convenience function to generate test cases."""
    generator = CaseGenerator(runtime, constraints)
    return generator.process(scenarios)