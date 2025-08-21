"""
Command Line Interface for QA Test Scenario Generator

Provides the main CLI entry point that matches the behavioral requirements:
- Single command that ingests inputs and writes artifacts
- Exit non-zero with machine-readable error if G1 fails  
- Print brief summary on success (counts + artifact path)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List
import logging

from .workflow import QAWorkflow, generate_qa_plan_from_files
from .runtime import RuntimeFactory, auto_detect_runtime
from .models import RequirementsInput, Constraints, APIInfo
from .exceptions import G1ValidationError, QAGeneratorError


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from some libraries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="QA Test Scenario Generator - Transform PRDs into comprehensive test plans using local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with files
  qa-generator --project payment-flow --artifact-id JIRA-123 \\
    --spec-file prd.txt --ac-file acceptance_criteria.json
  
  # With local MLX server  
  qa-generator --local-server http://localhost:8080/v1 \\
    --project checkout --artifact-id STORY-456 \\
    --spec-file spec.md --ac-file acs.json
  
  # With test framework skeletons
  qa-generator --project auth --artifact-id TASK-789 \\
    --spec-file requirements.txt --ac-file criteria.json \\
    --test-framework playwright --output-dir ./test_artifacts
  
  # Using hosted OpenAI API
  qa-generator --openai-api-key sk-... --project api \\
    --artifact-id FEAT-101 --spec-file api_spec.txt --ac-file acs.json

For more info: https://github.com/your-repo/qa-generator
        """
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--project',
        required=True,
        help='Project name for context'
    )
    required.add_argument(
        '--artifact-id', 
        required=True,
        help='Work item ID for traceability (e.g., JIRA-123, STORY-456)'
    )
    
    # Input specification  
    input_group = parser.add_argument_group('input specification')
    input_group.add_argument(
        '--spec-file',
        type=Path,
        help='Path to PRD/specification file'
    )
    input_group.add_argument(
        '--spec-text',
        help='Inline specification text (alternative to --spec-file)'
    )
    input_group.add_argument(
        '--ac-file',
        type=Path,
        help='Path to acceptance criteria JSON file'
    )
    input_group.add_argument(
        '--ac-json',
        help='Inline acceptance criteria as JSON string (alternative to --ac-file)'
    )
    
    # Optional API schema
    input_group.add_argument(
        '--api-schema',
        type=Path,
        help='Path to OpenAPI/GraphQL schema file (informational)'
    )
    
    # LLM Runtime options
    runtime_group = parser.add_argument_group('LLM runtime options')
    runtime_group.add_argument(
        '--local-server',
        help='Local MLX server URL (default: http://localhost:8080/v1)'
    )
    runtime_group.add_argument(
        '--openai-api-key',
        help='OpenAI API key for hosted inference'
    )
    runtime_group.add_argument(
        '--anthropic-api-key',
        help='Anthropic API key for hosted inference'
    )
    runtime_group.add_argument(
        '--prefer-local',
        action='store_true',
        default=True,
        help='Prefer local runtime over hosted (default: True)'
    )
    
    # Generation options
    gen_group = parser.add_argument_group('generation options')
    gen_group.add_argument(
        '--test-framework',
        choices=['playwright', 'selenium', 'pytest', 'cypress', 'jest'],
        help='Generate test skeletons for specified framework'
    )
    gen_group.add_argument(
        '--priority-policy',
        choices=['risk_weighted', 'uniform'],
        default='risk_weighted',
        help='Priority assignment policy (default: risk_weighted)'
    )
    gen_group.add_argument(
        '--environments',
        nargs='+',
        help='Target environments (e.g., staging prod mobile)'
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Output directory for artifacts (default: current directory)'
    )
    output_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate command line inputs."""
    
    # Validate specification input
    if not args.spec_file and not args.spec_text:
        raise ValueError("Either --spec-file or --spec-text is required")
    
    if args.spec_file and args.spec_text:
        raise ValueError("Provide either --spec-file or --spec-text, not both")
    
    # Validate AC input
    if not args.ac_file and not args.ac_json:
        raise ValueError("Either --ac-file or --ac-json is required")
    
    if args.ac_file and args.ac_json:
        raise ValueError("Provide either --ac-file or --ac-json, not both")
    
    # Validate file paths exist
    if args.spec_file and not args.spec_file.exists():
        raise FileNotFoundError(f"Specification file not found: {args.spec_file}")
    
    if args.ac_file and not args.ac_file.exists():
        raise FileNotFoundError(f"Acceptance criteria file not found: {args.ac_file}")
    
    if args.api_schema and not args.api_schema.exists():
        raise FileNotFoundError(f"API schema file not found: {args.api_schema}")


def load_requirements_input(args: argparse.Namespace) -> RequirementsInput:
    """Load and create RequirementsInput from command line arguments."""
    
    # Load specification text
    if args.spec_file:
        spec_text = args.spec_file.read_text(encoding='utf-8')
    else:
        spec_text = args.spec_text
    
    # Load acceptance criteria
    if args.ac_file:
        with args.ac_file.open('r', encoding='utf-8') as f:
            ac_data = json.load(f)
    else:
        ac_data = json.loads(args.ac_json)
    
    # Handle different AC formats
    if isinstance(ac_data, list):
        acceptance_criteria = ac_data
    elif isinstance(ac_data, dict) and "acceptance_criteria" in ac_data:
        acceptance_criteria = ac_data["acceptance_criteria"]
    else:
        raise ValueError("AC data must be a list of strings or object with 'acceptance_criteria' field")
    
    # Create API info if provided
    api_info = None
    if args.api_schema:
        api_info = APIInfo(path=str(args.api_schema))
    
    # Create constraints
    constraints = Constraints(
        test_framework=args.test_framework,
        environments=args.environments,
        priority_policy=args.priority_policy
    )
    
    return RequirementsInput(
        project=args.project,
        artifact_id=args.artifact_id,
        spec_text=spec_text,
        acceptance_criteria=acceptance_criteria,
        api=api_info,
        constraints=constraints
    )


def create_runtime(args: argparse.Namespace):
    """Create LLM runtime based on command line arguments."""
    
    # Determine API key for hosted services
    api_key = args.openai_api_key or args.anthropic_api_key
    
    # Create runtime with preferences
    runtime = auto_detect_runtime(
        local_url=args.local_server,
        api_key=api_key
    )
    
    return runtime


def print_success_summary(results: dict) -> None:
    """Print brief success summary to stdout."""
    
    stats = results["execution_stats"]
    artifacts = results["artifacts"]
    summary = artifacts["summary"]["statistics"]
    
    print(f"✅ QA Test Plan Generated Successfully")
    print(f"Project: {results['input'].project}")
    print(f"Artifact ID: {results['input'].artifact_id}")
    print(f"")
    print(f"📊 Summary:")
    print(f"  Acceptance Criteria: {summary['acceptance_criteria']}")
    print(f"  Test Scenarios: {summary['scenarios']}")
    print(f"  Test Cases: {summary['test_cases']}")
    print(f"  Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  Open Questions: {summary['open_questions']}")
    print(f"")
    print(f"📁 Artifacts:")
    print(f"  JSON Plan: {artifacts['json_plan_path']}")
    
    if artifacts["skeleton_paths"]:
        print(f"  Test Skeletons:")
        for filename in artifacts["skeleton_paths"]:
            print(f"    - {filename}")
    
    print(f"")
    print(f"🎯 Priority Distribution:")
    priority_dist = summary["priority_distribution"]
    for priority, count in priority_dist.items():
        if count > 0:
            print(f"  {priority}: {count} test cases")


def print_error_summary(error: Exception) -> None:
    """Print machine-readable error summary to stderr."""
    
    if isinstance(error, G1ValidationError):
        # Machine-readable G1 violation report
        error_report = {
            "error_type": "G1_VALIDATION_FAILURE",
            "violated_rules": error.violated_rules,
            "offending_ids": error.offending_ids,
            "details": error.details,
            "message": str(error)
        }
        
        print(json.dumps(error_report, indent=2), file=sys.stderr)
        
    else:
        # General error report
        error_report = {
            "error_type": type(error).__name__,
            "message": str(error),
            "details": getattr(error, 'details', None)
        }
        
        print(json.dumps(error_report, indent=2), file=sys.stderr)


def main() -> int:
    """Main CLI entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Load requirements
        requirements = load_requirements_input(args)
        
        # Create runtime
        runtime = create_runtime(args)
        
        # Create and execute workflow
        workflow = QAWorkflow(runtime, args.output_dir)
        results = workflow.run(requirements)
        
        # Print success summary
        print_success_summary(results)
        
        return 0
        
    except G1ValidationError as e:
        logger.error(f"G1 validation failed: {e}")
        print_error_summary(e)
        return 1
        
    except (QAGeneratorError, ValueError, FileNotFoundError) as e:
        logger.error(f"Input validation failed: {e}")
        print_error_summary(e)
        return 2
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130  # Standard Unix exit code for SIGINT
        
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print_error_summary(e)
        return 3


if __name__ == '__main__':
    sys.exit(main())