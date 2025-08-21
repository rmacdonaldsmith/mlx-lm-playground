"""
ScenarioSynthesizer Node (LLM-based)

Generates comprehensive test scenarios from parsed requirements.
Ensures every AC gets at least one happy path and one negative scenario.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

from ..runtime import LLMRuntime
from ..validation import generate_with_validation
from ..models import (
    ParsedRequirements,
    Scenario,
    ScenarioGenerationResponse,
    AcceptanceCriteria
)

logger = logging.getLogger(__name__)


class ScenarioSynthesizer:
    """
    Node 2: Generate test scenarios from parsed requirements using LLM.
    
    For each AC, generates:
    - At least 1 happy path scenario (positive flow)
    - At least 1 negative scenario (error/edge cases)
    - Additional boundary scenarios when patterns are detected
    """
    
    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime
    
    def process(self, parsed_requirements: ParsedRequirements) -> List[Scenario]:
        """
        Generate comprehensive test scenarios from parsed requirements.
        
        Args:
            parsed_requirements: Output from ParseRequirements node
            
        Returns:
            List of generated test scenarios
        """
        
        # Build the generation prompt
        prompt = self._build_scenario_prompt(parsed_requirements)
        
        # Generate scenarios using LLM with validation
        response = generate_with_validation(
            runtime=self.runtime,
            prompt=prompt,
            response_model=ScenarioGenerationResponse,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=3000
        )
        
        # Post-process scenarios to ensure requirements are met
        scenarios = self._post_process_scenarios(
            response.scenarios, 
            parsed_requirements.acceptance_criteria
        )
        
        logger.info(f"Generated {len(scenarios)} test scenarios")
        return scenarios
    
    def _build_scenario_prompt(self, parsed_requirements: ParsedRequirements) -> str:
        """Build the LLM prompt for scenario generation."""
        
        acs_text = "\n".join([
            f"- {ac.id}: {ac.text}" 
            for ac in parsed_requirements.acceptance_criteria
        ])
        
        entities_summary = parsed_requirements.entities.get("summary", "") if parsed_requirements.entities else ""
        
        # Build context about extracted entities
        context_parts = []
        if parsed_requirements.entities:
            if parsed_requirements.entities.get("fields"):
                context_parts.append(f"Key fields: {', '.join(parsed_requirements.entities['fields'][:5])}")
            if parsed_requirements.entities.get("data_types"):
                context_parts.append(f"Data types: {', '.join(parsed_requirements.entities['data_types'][:3])}")
            if parsed_requirements.entities.get("validations"):
                context_parts.append(f"Validations: {', '.join(parsed_requirements.entities['validations'][:3])}")
        
        context_str = ". ".join(context_parts) if context_parts else "Standard web application functionality"
        
        prompt = f"""You are a QA test scenario generator. Generate comprehensive test scenarios for the following acceptance criteria.

ACCEPTANCE CRITERIA:
{acs_text}

CONTEXT: {context_str}

REQUIREMENTS:
1. For each acceptance criterion, generate:
   - At least 1 HAPPY PATH scenario (normal positive flow)
   - At least 1 NEGATIVE scenario (error/failure case)
   - Additional BOUNDARY scenarios if relevant (edge cases, limits, format variations)

2. Scenario requirements:
   - Use IDs: SCN-001, SCN-002, SCN-003, etc.
   - Set type: "functional", "integration", or "e2e"
   - Set risk: "low", "medium", or "high" (consider data integrity, security, payments)
   - List related_requirements as AC IDs that this scenario covers
   - Include preconditions if setup is needed
   - Add variants for different contexts (locale, auth state, etc.) if relevant

3. Scenario types to consider:
   - Input validation scenarios (required fields, format validation, length limits)
   - Authentication/authorization scenarios
   - Error handling scenarios (network errors, server errors, invalid data)
   - Boundary testing (min/max values, edge cases)
   - Integration scenarios (API calls, database operations, third-party services)

4. Risk assessment guidelines:
   - HIGH: Payment processing, authentication, data privacy, security
   - MEDIUM: Data integrity, user experience, business logic
   - LOW: Cosmetic, informational, non-critical features

Generate scenarios that provide comprehensive coverage of both positive and negative test cases.

IMPORTANT: Return ONLY the JSON data structure below. NO explanations, NO schema definitions, NO markdown formatting.

{{
  "scenarios": [
    {{
      "id": "SCN-001",
      "title": "Descriptive scenario title",
      "description": "Detailed scenario description",
      "type": "functional",
      "risk": "medium",
      "preconditions": ["prerequisite 1", "prerequisite 2"],
      "related_requirements": ["AC-001"],
      "tags": ["validation", "positive"],
      "variants": []
    }}
  ]
}}"""
        
        return prompt
    
    def _post_process_scenarios(
        self, 
        scenarios: List[Scenario],
        acceptance_criteria: List[AcceptanceCriteria]
    ) -> List[Scenario]:
        """
        Post-process generated scenarios to ensure quality and completeness.
        
        Args:
            scenarios: Raw scenarios from LLM
            acceptance_criteria: Original ACs for validation
            
        Returns:
            Processed and validated scenarios
        """
        
        # Step 1: Validate scenario IDs and fix any issues
        scenarios = self._fix_scenario_ids(scenarios)
        
        # Step 2: Ensure every AC has both positive and negative coverage
        scenarios = self._ensure_ac_coverage(scenarios, acceptance_criteria)
        
        # Step 3: Validate AC references
        scenarios = self._validate_ac_references(scenarios, acceptance_criteria)
        
        return scenarios
    
    def _fix_scenario_ids(self, scenarios: List[Scenario]) -> List[Scenario]:
        """Ensure scenario IDs are properly formatted and unique."""
        fixed_scenarios = []
        used_ids = set()
        
        for i, scenario in enumerate(scenarios, 1):
            # Generate proper ID format
            scenario_id = f"SCN-{i:03d}"
            
            # Handle duplicates
            counter = 1
            original_id = scenario_id
            while scenario_id in used_ids:
                counter += 1
                scenario_id = f"{original_id[:-3]}{counter:03d}"
            
            used_ids.add(scenario_id)
            
            # Create new scenario with corrected ID
            fixed_scenario = Scenario(
                id=scenario_id,
                title=scenario.title,
                type=scenario.type,
                risk=scenario.risk,
                related_requirements=scenario.related_requirements,
                preconditions=scenario.preconditions,
                variants=scenario.variants
            )
            
            fixed_scenarios.append(fixed_scenario)
        
        return fixed_scenarios
    
    def _ensure_ac_coverage(
        self, 
        scenarios: List[Scenario],
        acceptance_criteria: List[AcceptanceCriteria]
    ) -> List[Scenario]:
        """
        Ensure every AC has both positive and negative scenario coverage.
        Add missing scenarios if needed.
        """
        
        # Analyze existing coverage
        ac_coverage = self._analyze_scenario_coverage(scenarios)
        
        additional_scenarios = []
        next_id = len(scenarios) + 1
        
        for ac in acceptance_criteria:
            ac_id = ac.id
            scenarios_for_ac = ac_coverage.get(ac_id, [])
            
            # Check if we have positive scenarios
            has_positive = any(
                self._is_likely_positive_scenario(s) for s in scenarios_for_ac
            )
            
            # Check if we have negative scenarios  
            has_negative = any(
                self._is_likely_negative_scenario(s) for s in scenarios_for_ac
            )
            
            # Add missing positive scenario
            if not has_positive:
                positive_scenario = self._create_default_positive_scenario(
                    ac, f"SCN-{next_id:03d}"
                )
                additional_scenarios.append(positive_scenario)
                next_id += 1
            
            # Add missing negative scenario
            if not has_negative:
                negative_scenario = self._create_default_negative_scenario(
                    ac, f"SCN-{next_id:03d}"
                )
                additional_scenarios.append(negative_scenario)
                next_id += 1
        
        if additional_scenarios:
            logger.info(f"Added {len(additional_scenarios)} scenarios for complete AC coverage")
        
        return scenarios + additional_scenarios
    
    def _analyze_scenario_coverage(self, scenarios: List[Scenario]) -> Dict[str, List[Scenario]]:
        """Build map of AC ID to scenarios that cover it."""
        coverage = {}
        
        for scenario in scenarios:
            for ac_id in scenario.related_requirements:
                if ac_id not in coverage:
                    coverage[ac_id] = []
                coverage[ac_id].append(scenario)
        
        return coverage
    
    def _is_likely_positive_scenario(self, scenario: Scenario) -> bool:
        """Heuristic to detect positive/happy path scenarios."""
        title_lower = scenario.title.lower()
        negative_indicators = [
            'invalid', 'error', 'fail', 'missing', 'empty', 'wrong',
            'unauthorized', 'forbidden', 'denied', 'timeout', 'limit'
        ]
        
        return not any(indicator in title_lower for indicator in negative_indicators)
    
    def _is_likely_negative_scenario(self, scenario: Scenario) -> bool:
        """Heuristic to detect negative/error scenarios."""
        return not self._is_likely_positive_scenario(scenario)
    
    def _create_default_positive_scenario(self, ac: AcceptanceCriteria, scenario_id: str) -> Scenario:
        """Create a default positive scenario for an AC."""
        return Scenario(
            id=scenario_id,
            title=f"Valid case for {ac.id}",
            type="functional",
            risk="medium",
            related_requirements=[ac.id],
            preconditions="System is in normal operating state",
            variants=None
        )
    
    def _create_default_negative_scenario(self, ac: AcceptanceCriteria, scenario_id: str) -> Scenario:
        """Create a default negative scenario for an AC."""
        return Scenario(
            id=scenario_id,
            title=f"Invalid case for {ac.id}",
            type="functional", 
            risk="medium",
            related_requirements=[ac.id],
            preconditions="System is in normal operating state",
            variants=None
        )
    
    def _validate_ac_references(
        self, 
        scenarios: List[Scenario],
        acceptance_criteria: List[AcceptanceCriteria]
    ) -> List[Scenario]:
        """Validate that all AC references in scenarios are valid."""
        
        valid_ac_ids = {ac.id for ac in acceptance_criteria}
        cleaned_scenarios = []
        
        for scenario in scenarios:
            # Filter out invalid AC references
            valid_requirements = [
                ac_id for ac_id in scenario.related_requirements
                if ac_id in valid_ac_ids
            ]
            
            # Skip scenarios with no valid AC references
            if not valid_requirements:
                logger.warning(f"Scenario {scenario.id} has no valid AC references, skipping")
                continue
            
            # Create scenario with cleaned AC references
            cleaned_scenario = Scenario(
                id=scenario.id,
                title=scenario.title,
                type=scenario.type,
                risk=scenario.risk,
                related_requirements=valid_requirements,
                preconditions=scenario.preconditions,
                variants=scenario.variants
            )
            
            cleaned_scenarios.append(cleaned_scenario)
        
        return cleaned_scenarios


# Convenience function

def synthesize_scenarios(
    parsed_requirements: ParsedRequirements, 
    runtime: LLMRuntime
) -> List[Scenario]:
    """Convenience function to generate scenarios."""
    synthesizer = ScenarioSynthesizer(runtime)
    return synthesizer.process(parsed_requirements)