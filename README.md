# MLX QA Generator

Transform Product Requirements Documents (PRDs) and acceptance criteria into comprehensive test plans using local MLX models.

## 🎯 Overview

The MLX QA Generator is a 5-node workflow system that uses local Apple MLX models to generate comprehensive test scenarios, test cases, and validation artifacts from product requirements. Built with a local-first approach to avoid API costs while maintaining enterprise-grade quality.

## ✨ Key Features

### 🔄 5-Node Workflow Architecture
1. **ParseRequirements** - Normalizes PRDs and acceptance criteria
2. **ScenarioSynthesizer** - Generates comprehensive test scenarios using LLM
3. **CaseGenerator** - Expands scenarios into detailed test cases with steps
4. **CoverageCritic** - Validates completeness using G1 quality gates
5. **ArtifactEmitter** - Outputs JSON and optional test framework skeletons

### 🏠 Local-First LLM Runtime
- Uses quantized MLX models (Apple Silicon optimized)
- No API costs for default operation
- Automatic fallback to hosted APIs (OpenAI, Anthropic)
- Pluggable runtime system supports both local and cloud models

### 🎯 Quality Assurance
- **G1 Quality Gates** (G1.1-G1.5) ensure comprehensive test coverage
- JSON schema validation with automatic retry logic
- Deterministic output using low temperature settings
- Comprehensive error handling and logging

### 🧪 Multi-Framework Support
Generate test skeletons for:
- Playwright
- Selenium
- Pytest
- Cypress
- Jest

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/rmacdonaldsmith/mlx-lm-playground.git
cd mlx-lm-playground
pip install -e .
```

### Basic Usage

```bash
# Generate test scenarios from PRD and acceptance criteria
qa-generator \
  --project payment-flow \
  --artifact-id JIRA-123 \
  --spec-file requirements.txt \
  --ac-file acceptance_criteria.json \
  --output-dir ./test_artifacts
```

### With Test Framework Skeletons

```bash
# Generate Playwright test skeletons
qa-generator \
  --project checkout-flow \
  --artifact-id STORY-456 \
  --spec-file prd.md \
  --ac-file criteria.json \
  --test-framework playwright \
  --output-dir ./tests
```

## 📋 Input Requirements

### Requirements File (--spec-file)
Detailed product requirements with:
- Feature descriptions
- Business rules
- Field validation requirements
- User interaction flows

### Acceptance Criteria JSON (--ac-file)
```json
{
  "acceptance_criteria": [
    "US ZIP code is required when saving a card",
    "Card number must pass Luhn validation; otherwise show a validation error"
  ]
}
```

## 📊 Output Artifacts

### Test Scenarios JSON
```json
{
  "scenarios": [
    {
      "id": "SCN-001",
      "title": "Valid ZIP code entry",
      "description": "User enters valid US ZIP code format",
      "type": "functional",
      "risk": "medium",
      "preconditions": ["User on payment form"],
      "related_requirements": ["AC-001"],
      "tags": ["validation", "positive"]
    }
  ]
}
```

### Test Cases with Steps
```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "scenario_id": "SCN-001", 
      "title": "Enter valid 5-digit ZIP code",
      "steps": [
        {"action": "Navigate to payment form", "expected": "Form displays"},
        {"action": "Enter '12345' in ZIP field", "expected": "Field accepts input"},
        {"action": "Tab to next field", "expected": "No validation error shown"}
      ],
      "priority": "high",
      "environment": "staging"
    }
  ]
}
```

## 🛠️ Configuration

### LLM Runtime Options

```bash
# Use local MLX model (default)
qa-generator --prefer-local ...

# Use OpenAI API
qa-generator --openai-api-key sk-... ...

# Use custom local server
qa-generator --local-server http://localhost:8080/v1 ...
```

### Generation Options

```bash
# Set priority policy
qa-generator --priority-policy risk_weighted ...

# Target specific environments  
qa-generator --environments staging prod mobile ...

# Enable verbose logging
qa-generator --verbose ...
```

## 🏗️ Architecture

### Runtime System
- **SimpleMLXRuntime**: Direct in-process MLX usage with proper temperature control
- **OpenAICompatibleRuntime**: Unified interface for hosted APIs
- **RuntimeFactory**: Auto-detection and fallback management

### Validation System
- JSON schema validation using Pydantic v2
- Automatic repair strategies for common LLM output issues
- Retry logic with enhanced prompts
- Comprehensive error reporting

### Node System
Each workflow node is independently testable with clear interfaces:
```python
class WorkflowNode(Protocol):
    def process(self, input_data: T) -> U:
        """Process input and return validated output."""
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_models.py -v

# Test with real MLX model (slow)
pytest tests/test_zip_luhn_example.py -v
```

## 📝 Examples

See `examples/` directory for complete working examples:

```bash
# Run payment form example
./examples/run_payment_example.sh
```

## 🔧 Development

### Project Structure
```
qa_generator/
├── __init__.py           # Main exports
├── cli.py               # Command-line interface  
├── models.py            # Pydantic data models
├── runtime.py           # LLM runtime abstraction
├── simple_mlx_runtime.py # Direct MLX implementation
├── validation.py        # JSON validation & repair
├── workflow.py          # Main workflow orchestration
└── nodes/              # Individual workflow nodes
    ├── parser.py        # Requirements parsing
    ├── synthesizer.py   # Scenario generation  
    ├── generator.py     # Test case generation
    ├── critic.py        # Coverage validation
    └── emitter.py       # Artifact output
```

### Key Dependencies
- `mlx` + `mlx-lm`: Local model inference
- `pydantic>=2.5`: Data validation and schemas
- `openai>=1.0.0`: Unified API client
- `requests`: HTTP communication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Apple MLX team for the excellent local inference framework
- OpenAI for the API compatibility standards
- Pydantic team for robust data validation