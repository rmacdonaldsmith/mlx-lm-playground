"""
QA Test Scenario Generator Workflow

Orchestrates the 5-node workflow:
1. ParseRequirements → 2. ScenarioSynthesizer → 3. CaseGenerator 
→ 4. CoverageCritic → 5. ArtifactEmitter

Handles the complete end-to-end process with error handling and logging.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .runtime import LLMRuntime, RuntimeFactory
from .models import RequirementsInput, TestPlan, Constraints
from .nodes import (
    ParseRequirements,
    ScenarioSynthesizer, 
    CaseGenerator,
    CoverageCritic,
    ArtifactEmitter
)
from .exceptions import G1ValidationError, QAGeneratorError

logger = logging.getLogger(__name__)


class QAWorkflow:
    """
    Main workflow orchestrator for QA Test Scenario Generator.
    
    Executes the complete 5-node pipeline with proper error handling,
    logging, and artifact generation.
    """
    
    def __init__(
        self,
        runtime: Optional[LLMRuntime] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            runtime: LLM runtime to use (auto-detected if None)
            output_dir: Output directory for artifacts (current dir if None)
        """
        self.runtime = runtime
        self.output_dir = output_dir or Path.cwd()
        
        # Will be populated during execution
        self._execution_stats = {}
    
    def run(self, requirements: RequirementsInput) -> Dict[str, Any]:
        """
        Execute the complete QA generation workflow.
        
        Args:
            requirements: Input requirements and acceptance criteria
            
        Returns:
            Dict containing generated artifacts and execution statistics
            
        Raises:
            G1ValidationError: If quality gates fail
            QAGeneratorError: If any workflow step fails
        """
        
        logger.info(f"Starting QA generation workflow for project: {requirements.project}")
        
        try:
            # Initialize runtime if not provided
            if not self.runtime:
                self.runtime = self._initialize_runtime()
            
            # Execute 5-node workflow
            results = self._execute_workflow(requirements)
            
            # Log success summary
            self._log_success_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise
    
    def _initialize_runtime(self) -> LLMRuntime:
        """Initialize LLM runtime with auto-detection."""
        logger.info("Auto-detecting LLM runtime...")
        
        factory = RuntimeFactory()
        runtime = factory.create_runtime(auto_fallback=True)
        
        info = runtime.get_model_info()
        logger.info(f"Using runtime: {info['name']} ({info['type']})")
        
        return runtime
    
    def _execute_workflow(self, requirements: RequirementsInput) -> Dict[str, Any]:
        """Execute the 5-node workflow pipeline."""
        
        # Node 1: ParseRequirements (deterministic)
        logger.info("Node 1: Parsing requirements...")
        parsed_requirements = ParseRequirements.process(requirements)
        
        self._execution_stats["node1_acs"] = len(parsed_requirements.acceptance_criteria)
        logger.info(f"Normalized {len(parsed_requirements.acceptance_criteria)} acceptance criteria")
        
        # Node 2: ScenarioSynthesizer (LLM-based)
        logger.info("Node 2: Generating test scenarios...")
        synthesizer = ScenarioSynthesizer(self.runtime)
        scenarios = synthesizer.process(parsed_requirements)
        
        self._execution_stats["node2_scenarios"] = len(scenarios)
        logger.info(f"Generated {len(scenarios)} test scenarios")
        
        # Node 3: CaseGenerator (LLM-based)
        logger.info("Node 3: Generating test cases...")
        generator = CaseGenerator(self.runtime, parsed_requirements.constraints)
        test_cases = generator.process(scenarios)
        
        self._execution_stats["node3_test_cases"] = len(test_cases)
        logger.info(f"Generated {len(test_cases)} test cases")
        
        # Node 4: CoverageCritic (deterministic validation)
        logger.info("Node 4: Validating coverage and quality gates...")
        critic = CoverageCritic(self.runtime)
        coverage_map, open_questions = critic.process(
            parsed_requirements.acceptance_criteria,
            scenarios,
            test_cases
        )
        
        self._execution_stats["node4_questions"] = len(open_questions)
        self._execution_stats["node4_coverage"] = len(coverage_map.ac_to_scenarios)
        logger.info(f"Coverage analysis: {len(coverage_map.ac_to_scenarios)} ACs covered, "
                   f"{len(open_questions)} open questions")
        
        # Node 5: ArtifactEmitter (deterministic output)
        logger.info("Node 5: Emitting final artifacts...")
        emitter = ArtifactEmitter(self.output_dir)
        artifacts = emitter.process(
            requirements,
            parsed_requirements.acceptance_criteria,
            scenarios,
            test_cases,
            coverage_map,
            open_questions,
            parsed_requirements.constraints
        )
        
        # Combine all results
        results = {
            "input": requirements,
            "parsed_requirements": parsed_requirements,
            "scenarios": scenarios,
            "test_cases": test_cases,
            "coverage_map": coverage_map,
            "open_questions": open_questions,
            "artifacts": artifacts,
            "execution_stats": self._execution_stats,
            "test_plan": artifacts["test_plan"]  # For easy access
        }
        
        return results
    
    def _log_success_summary(self, results: Dict[str, Any]) -> None:
        """Log successful execution summary."""
        
        stats = results["execution_stats"]
        artifacts = results["artifacts"]
        
        summary_lines = [
            "",
            "🎉 QA Test Scenario Generation Complete!",
            "=" * 50,
            f"Project: {results['input'].project}",
            f"Artifact ID: {results['input'].artifact_id}",
            "",
            "📊 Generation Statistics:",
            f"  • Acceptance Criteria: {stats.get('node1_acs', 0)}",
            f"  • Test Scenarios: {stats.get('node2_scenarios', 0)}",
            f"  • Test Cases: {stats.get('node3_test_cases', 0)}",
            f"  • Open Questions: {stats.get('node4_questions', 0)}",
            f"  • Coverage: {stats.get('node4_coverage', 0)} ACs covered",
            "",
            "📁 Generated Artifacts:",
            f"  • JSON Plan: {artifacts['json_plan_path']}",
        ]
        
        if artifacts["skeleton_paths"]:
            summary_lines.append("  • Test Skeletons:")
            for name, path in artifacts["skeleton_paths"].items():
                summary_lines.append(f"    - {name}: {path}")
        
        summary_lines.extend([
            "",
            "✅ All G1 quality gates passed",
            "=" * 50,
            ""
        ])
        
        for line in summary_lines:
            logger.info(line)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._execution_stats.copy()


# Convenience functions for common workflows

def generate_qa_plan(
    requirements: RequirementsInput,
    runtime: Optional[LLMRuntime] = None,
    output_dir: Optional[Path] = None
) -> TestPlan:
    """
    Convenience function to generate QA test plan.
    
    Args:
        requirements: Input requirements
        runtime: Optional LLM runtime (auto-detected if None)
        output_dir: Output directory (current dir if None)
        
    Returns:
        Generated test plan
    """
    workflow = QAWorkflow(runtime, output_dir)
    results = workflow.run(requirements)
    return results["test_plan"]


def generate_qa_plan_from_files(
    project: str,
    artifact_id: str,
    spec_file: Path,
    ac_file: Path,
    constraints: Optional[Constraints] = None,
    runtime: Optional[LLMRuntime] = None,
    output_dir: Optional[Path] = None
) -> TestPlan:
    """
    Generate QA test plan from specification files.
    
    Args:
        project: Project name
        artifact_id: Artifact/work item ID
        spec_file: Path to specification/PRD file
        ac_file: Path to acceptance criteria file (JSON)
        constraints: Optional generation constraints
        runtime: Optional LLM runtime
        output_dir: Output directory
        
    Returns:
        Generated test plan
    """
    import json
    
    # Read specification text
    spec_text = spec_file.read_text(encoding='utf-8')
    
    # Read acceptance criteria  
    with ac_file.open('r', encoding='utf-8') as f:
        ac_data = json.load(f)
    
    # Handle different AC file formats
    if isinstance(ac_data, list):
        acceptance_criteria = ac_data
    elif isinstance(ac_data, dict) and "acceptance_criteria" in ac_data:
        acceptance_criteria = ac_data["acceptance_criteria"]
    else:
        raise QAGeneratorError(f"Invalid AC file format: {ac_file}")
    
    # Create requirements input
    requirements = RequirementsInput(
        project=project,
        artifact_id=artifact_id,
        spec_text=spec_text,
        acceptance_criteria=acceptance_criteria,
        constraints=constraints
    )
    
    return generate_qa_plan(requirements, runtime, output_dir)