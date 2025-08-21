#!/bin/bash

# Payment Example - Generate QA Test Plan
# This script demonstrates the QA generator with a realistic payment form scenario

set -e

echo "🚀 MLX QA Suite - Payment Form Example"
echo "======================================"
echo

# Check if qa-generator is available
if ! command -v qa-generator &> /dev/null; then
    echo "❌ qa-generator not found. Please install the package first:"
    echo "   pip install -e ."
    exit 1
fi

# Check if mlx-llm-server is running
echo "🔍 Checking for local MLX server..."
if curl -s http://localhost:8080/health &> /dev/null; then
    echo "✅ Local MLX server is running at http://localhost:8080"
    SERVER_ARG="--local-server http://localhost:8080/v1"
elif [[ -n "$OPENAI_API_KEY" ]]; then
    echo "🌐 Using OpenAI API (OPENAI_API_KEY found)"
    SERVER_ARG="--openai-api-key $OPENAI_API_KEY"
else
    echo "⚠️  No local MLX server detected and no OPENAI_API_KEY set"
    echo "   Please either:"
    echo "   1. Start MLX server: python -m mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"
    echo "   2. Set OPENAI_API_KEY environment variable"
    echo "   3. Continue anyway and let the system auto-detect runtime"
    SERVER_ARG=""
fi

# Create output directory
OUTPUT_DIR="./examples/payment_output"
mkdir -p "$OUTPUT_DIR"

echo
echo "📝 Generating QA test plan for payment form..."
echo "   Project: payment-processing"
echo "   Artifact: PAY-001"  
echo "   Framework: playwright"
echo "   Output: $OUTPUT_DIR"
echo

# Run the QA generator
qa-generator \
  --project "payment-processing" \
  --artifact-id "PAY-001" \
  --spec-file "./examples/payment_example/requirements.txt" \
  --ac-file "./examples/payment_example/acceptance_criteria.json" \
  --test-framework "playwright" \
  --priority-policy "risk_weighted" \
  --environments "staging" "production" "mobile" \
  --output-dir "$OUTPUT_DIR" \
  --verbose \
  $SERVER_ARG

echo
echo "🎉 Generation Complete!"
echo "======================================"
echo "📁 Generated Artifacts:"
echo "   • JSON Plan: $OUTPUT_DIR/qa_test_plan_PAY-001.json"
echo "   • Test Skeletons: $OUTPUT_DIR/test_skeletons_PAY-001/"
echo

# Show quick summary
if [[ -f "$OUTPUT_DIR/qa_test_plan_PAY-001.json" ]]; then
    echo "📊 Quick Summary:"
    echo -n "   • Scenarios: "
    jq -r '.scenarios | length' "$OUTPUT_DIR/qa_test_plan_PAY-001.json" 2>/dev/null || echo "N/A"
    echo -n "   • Test Cases: "
    jq -r '.test_cases | length' "$OUTPUT_DIR/qa_test_plan_PAY-001.json" 2>/dev/null || echo "N/A"
    echo -n "   • Open Questions: "
    jq -r '.open_questions | length' "$OUTPUT_DIR/qa_test_plan_PAY-001.json" 2>/dev/null || echo "N/A"
    echo -n "   • Coverage: "
    jq -r '.metadata.coverage_stats.acs_covered' "$OUTPUT_DIR/qa_test_plan_PAY-001.json" 2>/dev/null || echo "N/A"
    echo " ACs covered"
fi

echo
echo "🔍 Next Steps:"
echo "   1. Review generated test plan:"
echo "      cat $OUTPUT_DIR/qa_test_plan_PAY-001.json | jq ."
echo
echo "   2. Examine test skeletons:"
echo "      ls -la $OUTPUT_DIR/test_skeletons_PAY-001/"
echo
echo "   3. Run generated tests (after implementation):"
echo "      cd $OUTPUT_DIR/test_skeletons_PAY-001/ && playwright test"
echo
echo "✨ Happy Testing!"