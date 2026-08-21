"""
LLM Reasoning / Synthesis Node
==============================

Uses Groq openai/gpt-oss-120b to synthesize already-verified evidence.

Important grounding rule:
    The LLM may interpret the supplied evidence, but it must not
    invent numbers, forecasts, model results, sources, or business
    facts that are absent from the state.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.ai_layer.llm import get_reasoning_llm
from src.ai_layer.state import AgentState


SYSTEM_PROMPT = """
You are the reasoning engine for an Intelligent Forecasting Agent.

Analyze ONLY the verified project evidence supplied to you.

GROUNDING RULES
---------------
1. Never invent numerical values, forecasts, metrics, dates, or sources.
2. Never invent business facts or external industry benchmarks.
3. Treat project-defined thresholds, scores, and rules as project design
   choices, not industry standards.
4. Distinguish clearly between:
   - production forecast outputs,
   - historical validation results,
   - TreeSHAP explanations,
   - monitoring signals,
   - retrieved project context.
5. If evidence is missing, explicitly state that it is missing.
6. SHAP values are additive model contributions in model-output units.
   They are NOT percentages.
7. Risk scores must be written as "X/100", never "X%".
8. For quantitative comparisons, state the actual values instead of using
   unsupported qualitative descriptions such as "large", "unusually wide",
   "sharp", or "substantial" unless a project-defined status or threshold
   explicitly supports that wording.
9. Do not describe historical weighted-ensemble validation results as
   production XGBoost validation results.
10. Do not describe a production XGBoost forecast as a weighted-ensemble
    forecast.
11. Recommendations must be tied to supplied evidence.
12. When a recommendation is a possible/project-defined action rather than
    a direct conclusion from the evidence, label it explicitly as such.
13. Do not expose hidden chain-of-thought. Provide conclusions and concise
    supporting evidence only.

RESPONSE REQUIREMENTS
---------------------
Answer the user's business question directly.

Return:
- Synthesis
- Key findings
- Recommendations

Keep the response factual, concise, and traceable to the supplied evidence.
"""


def _serialize_evidence(
    state: AgentState,
) -> str:
    """Serialize the collected evidence into a compact JSON string."""

    evidence = {
        "user_query":
            state.get(
                "user_query",
                "",
            ),
        "forecast":
            state.get(
                "forecast"
            ),
        "shap_explanation":
            state.get(
                "shap_explanation"
            ),
        "historical_data":
            state.get(
                "historical_data"
            ),
        "business_context":
            state.get(
                "business_context"
            ),
        "risk_assessment":
            state.get(
                "risk_assessment"
            ),
    }

    return json.dumps(
        evidence,
        indent=2,
        default=str,
    )


def reason(
    state: AgentState,
) -> dict:
    """Generate grounded synthesis using Groq."""

    llm = get_reasoning_llm()

    evidence = _serialize_evidence(
        state
    )

    user_prompt = f"""
User question:
{state.get('user_query', '')}

Verified project evidence:
{evidence}

Using only the evidence above, produce:

1. Synthesis:
   Directly answer the user's question in 3-6 concise sentences.

2. Key findings:
   2-5 bullet points containing only evidence-supported findings.

3. Recommendations:
   1-3 practical recommendations.
   Every recommendation must either:
   - directly follow from the supplied evidence, or
   - be explicitly labeled as a possible/project-defined action.

Use exact numerical values when relevant.

Important terminology:
- Write risk scores as "X/100", not "X%".
- Write SHAP contributions as numeric model-output contributions,
  not percentages.
- Do not invent missing information.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]
    )

    content = (
        response.content
        if isinstance(
            response.content,
            str,
        )
        else str(
            response.content
        )
    )

    return {
        "synthesis":
            content
    }