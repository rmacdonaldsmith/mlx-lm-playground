"""
QA Test Scenario Generator

Transforms PRDs and acceptance criteria into comprehensive test plans using local LLMs.
Features deterministic JSON generation, schema validation, and OpenAI-compatible runtime.
"""

__version__ = "0.1.0"
__all__ = [
    "QAWorkflow", 
    "RequirementsInput", 
    "TestPlan", 
    "G1ValidationError"
]

from .models import RequirementsInput, TestPlan
from .workflow import QAWorkflow
from .exceptions import G1ValidationError