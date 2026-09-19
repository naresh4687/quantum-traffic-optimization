"""Featherless AI explanation layer: explains validated results; never controls the traffic system.

Public surface (nothing else in the project imports the SDK or the endpoint):

    from qtraffic.ai import (FeatherlessConfig, TrafficAnalysisContext, Explainer,
                             explain_optimization, explain_qaoa, explain_emergency,
                             explain_environment, explain_comparison, explain_signal_choice)
"""

from .config import FeatherlessConfig
from .context import ContextError, TrafficAnalysisContext
from .explain import (
    ExplanationResult, Explainer, deterministic_explanation, explain_comparison, explain_emergency,
    explain_environment, explain_optimization, explain_qaoa, explain_signal_choice, validate_llm_text,
)
from .featherless import FeatherlessError
from .results import load_controller_context, load_emergency_context, load_qaoa_context

__all__ = [
    "ContextError", "Explainer", "ExplanationResult", "FeatherlessConfig", "FeatherlessError",
    "TrafficAnalysisContext", "deterministic_explanation", "explain_comparison", "explain_emergency",
    "explain_environment", "explain_optimization", "explain_qaoa", "explain_signal_choice",
    "load_controller_context", "load_emergency_context", "load_qaoa_context", "validate_llm_text",
]
