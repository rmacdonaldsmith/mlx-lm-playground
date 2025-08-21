# MLX QA Suite

A comprehensive suite for local LLM development and QA test generation, featuring:

- **MLXChat**: Interactive chat interface for local MLX models
- **QA Generator**: Transforms PRDs into comprehensive test plans using local LLMs

## Features

### 🚀 Local-First LLM Runtime
- Uses quantized MLX models for cost-effective inference
- OpenAI-compatible HTTP interface via `mlx-llm-server`
- Automatic fallback to hosted APIs (OpenAI, Anthropic) if needed
- No API costs for default local operation

### 🧪 QA Test Scenario Generator
- **5-Node Workflow**: Parse → Synthesize → Generate → Validate → Emit
- **G1 Quality Gates**: Strict validation ensures comprehensive coverage
- **Deterministic Output**: Low temperature, JSON schemas, retry logic
- **Multi-Framework Support**: Generates test skeletons for Playwright, Pytest, Selenium, Cypress
- **Privacy-Aware**: All processing happens locally by default

## Quick Start

### Prerequisites

```bash
# Install MLX (Apple Silicon only)
pip install mlx mlx-lm

# Install the QA suite
pip install -e .

# Note: mlx-lm includes the server module, no separate package needed
```

### Basic Usage

1. **Start Local MLX Model Server**
```bash
# Terminal 1: Start MLX model server
python -m mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-3bit

# The server runs at http://localhost:8080 by default
```

2. **Generate QA Test Plan**
```bash
# Terminal 2: Generate comprehensive test plan
qa-generator --project payment-flow --artifact-id JIRA-123 \
  --spec-file requirements.txt \
  --ac-file acceptance_criteria.json \
  --test-framework playwright \
  --output-dir ./test_artifacts
```

3. **Review Generated Artifacts**
```bash
# JSON test plan
cat test_artifacts/qa_test_plan_JIRA-123.json

# Test skeletons (if --test-framework specified)
ls test_artifacts/test_skeletons_JIRA-123/
```

## Usage Examples

### Example 1: Payment Form Validation

**requirements.txt**
```
Payment form for credit card processing with ZIP code validation 
and Luhn algorithm verification. Users must enter valid US ZIP codes
and credit card numbers that pass Luhn validation.
```

**acceptance_criteria.json**
```json
{
  "acceptance_criteria": [
    "US ZIP code is required when saving a card",
    "Card number must pass Luhn; otherwise show a validation error"
  ]
}
```

**Command**
```bash
qa-generator --project payment-system --artifact-id PAY-001 \
  --spec-file requirements.txt --ac-file acceptance_criteria.json \
  --test-framework playwright --priority-policy risk_weighted
```

**Generated Output**
- ✅ JSON test plan with 5 scenarios, 12 test cases
- ✅ Playwright test skeletons
- ✅ 100% AC coverage (positive + negative cases)
- ✅ Open questions about international postal codes

### Example 2: Using Hosted APIs

```bash
# Use OpenAI instead of local MLX (no server needed)
qa-generator --openai-api-key sk-... \
  --project auth-system --artifact-id AUTH-456 \
  --spec-text "User login with email/password validation" \
  --ac-json '["Email format must be valid", "Password minimum 8 characters"]'
```

### Example 3: Inline Input

```bash
# No files needed - specify everything via CLI
qa-generator --project api-validation --artifact-id API-789 \
  --spec-text "REST API that accepts JSON payloads with validation" \
  --ac-json '["Request must be valid JSON", "Required fields must be present"]' \
  --test-framework pytest --environments staging prod
```

## Configuration

### Runtime Selection Priority
1. `--local-server http://localhost:8080/v1` (default if server running)
2. `--openai-api-key sk-...` (OpenAI hosted API)
3. `--anthropic-api-key sk-ant-...` (Anthropic hosted API)
4. Auto-detection with fallback

### Priority Policies
- **risk_weighted** (default): Higher risk → higher priority (P0/P1)
- **uniform**: Distribute priorities evenly (mostly P1/P2)

### Test Frameworks
- **playwright**: Browser automation with Playwright
- **selenium**: WebDriver-based browser testing  
- **pytest**: Python unit/integration testing
- **cypress**: JavaScript E2E testing
- **jest**: React/JavaScript testing

## Quality Gates (G1)

The system enforces strict quality gates to ensure comprehensive test coverage:

- **G1.1**: Every acceptance criterion has ≥1 scenario
- **G1.2**: Every scenario has ≥1 test case
- **G1.3**: Every AC has ≥1 positive AND ≥1 negative test case
- **G1.4**: All IDs are unique with valid references
- **G1.5**: All artifacts are valid JSON

If any gate fails, the system exits with status code 1 and a machine-readable error report.

## Output Structure

### JSON Test Plan
```json
{
  "project": "payment-system",
  "artifact_id": "PAY-001", 
  "acceptance_criteria": [...],
  "scenarios": [...],
  "test_cases": [...],
  "coverage_map": {...},
  "open_questions": [...],
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "total_scenarios": 5,
    "total_test_cases": 12,
    "coverage_stats": {...}
  }
}
```

### Test Skeletons
Generated code in your chosen framework:
```python
# test_generated.py (Playwright example)
def test_tc_001_valid_zip_provided(page: Page):
    """Test Case: TC-001"""
    # Step 1: Navigate to payment form
    # page.goto("payment-form-url")
    
    # Step 2: Enter valid ZIP code '12345'  
    # page.fill("#zip-code", "12345")
    
    # Expected: Form submits successfully
    # expect(page.locator(".success")).to_be_visible()
```

## CLI Reference

### Required Arguments
- `--project`: Project name for context
- `--artifact-id`: Work item ID (JIRA-123, STORY-456, etc.)

### Input Specification (choose one)
- `--spec-file path/to/spec.txt` OR `--spec-text "inline text"`
- `--ac-file path/to/criteria.json` OR `--ac-json '["criterion1", "criterion2"]'`

### Runtime Options
- `--local-server URL`: Local MLX server (default: http://localhost:8080/v1)
- `--openai-api-key KEY`: Use OpenAI hosted API
- `--anthropic-api-key KEY`: Use Anthropic hosted API
- `--prefer-local`: Prefer local over hosted (default: true)

### Generation Options
- `--test-framework {playwright,selenium,pytest,cypress,jest}`: Generate test skeletons
- `--priority-policy {risk_weighted,uniform}`: Priority assignment (default: risk_weighted)
- `--environments ENV1 ENV2`: Target environments

### Output Options
- `--output-dir DIR`: Output directory (default: current)
- `--verbose`: Enable debug logging

## Development

### Setup Development Environment
```bash
git clone <repo>
cd mlx-lm-playground
pip install -e ".[dev]"

# Start MLX server for testing
python -m mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-3bit
```

### Run Tests
```bash
# Unit tests (fast, no LLM calls)
pytest tests/test_models.py tests/test_parser.py tests/test_critic.py

# Integration tests (requires mock runtime)
pytest tests/test_zip_luhn_example.py

# All tests
pytest
```

### Code Quality
```bash
# Format code
black qa_generator tests

# Lint code  
ruff check qa_generator tests

# Type checking
mypy qa_generator
```

## Architecture

### 5-Node Workflow
1. **ParseRequirements** (deterministic): Normalize ACs, extract entities
2. **ScenarioSynthesizer** (LLM): Generate test scenarios with happy/negative paths
3. **CaseGenerator** (LLM): Expand scenarios to detailed test cases
4. **CoverageCritic** (deterministic): Validate G1 gates, generate questions
5. **ArtifactEmitter** (deterministic): Output JSON + test skeletons

### Runtime Abstraction
- **UnifiedRuntime**: OpenAI-compatible client for all LLM providers
- **MockRuntime**: Deterministic responses for testing
- **Auto-detection**: Graceful fallback from local to hosted

### Validation Strategy
- **Schema-driven**: Pydantic models with strict validation
- **Retry Logic**: Automatic JSON repair and retry on LLM errors
- **Deterministic**: Low temperature, structured prompts, stable output

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/mlx-qa-suite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/mlx-qa-suite/discussions)
- **Documentation**: This README + inline code documentation

---

**Made with ❤️ for the local LLM community**