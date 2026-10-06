"""
Agent orchestration for forecast-driven option valuation.

    [Market Data Agent] --(cleans & aligns)--> [schema gate] --> [Transformer Models]
        --(predictions)--> [Greeks/Validation Agent] --> [Final Valuation]

The two agents are LangChain tool-calling agents; the gate, the transformer
step and the final valuation are deterministic LangGraph nodes. The gate
re-validates the market data agent's panel independently and sends errors back
to the agent (up to ``max_data_attempts``) before the models ever run.

Usage (from ``pipeline/``):
    python orchestration.py "Value near-the-money AAPL options over the next 10 trading days"
Requires ANTHROPIC_API_KEY (and FRED_API_KEY for macro features) in the env or .env.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.agent_tools import RunContext, greeks_tools, market_data_tools, run_valuation, write_artifacts
from utils.forecasting import run_forecasts
from utils.prompts import GREEKS_VALIDATION_PROMPT, MARKET_DATA_PROMPT
from utils.schemas import ForecastConfig, PanelValidationError, validate_feature_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "claude-opus-5"
AGENT_RECURSION_LIMIT = 30


class PipelineState(TypedDict, total=False):
    query: str
    data_attempts: int
    gate_errors: list[str]
    panel_report: dict
    validation: dict
    summary: dict


def make_llm(model=DEFAULT_MODEL):
    # Server-side refusal fallback: a declined request is retried on another model.
    return ChatAnthropic(model=model, max_tokens=16000, betas=["server-side-fallback-2026-07-01"],
                         model_kwargs={"extra_body": {"fallbacks": "default"}})


def _last_text(result):
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def build_graph(ctx: RunContext, llm, max_data_attempts=3, allow_code=True, log=print):
    market_agent = create_agent(llm, market_data_tools(ctx), system_prompt=MARKET_DATA_PROMPT,
                                name="market_data_agent")
    greeks_agent = create_agent(llm, greeks_tools(ctx, allow_code=allow_code),
                                system_prompt=GREEKS_VALIDATION_PROMPT, name="greeks_validation_agent")
    run_cfg = {"recursion_limit": AGENT_RECURSION_LIMIT}

    def market_data(state: PipelineState):
        msg = state["query"]
        if state.get("gate_errors"):
            msg += "\n\nThe previous panel was rejected:\n- " + "\n- ".join(state["gate_errors"])
        result = market_agent.invoke({"messages": [{"role": "user", "content": msg}]}, run_cfg)
        log(f"[market_data] {_last_text(result)}")
        return {"data_attempts": state.get("data_attempts", 0) + 1}

    def schema_gate(state: PipelineState):
        # Independent of whatever the agent claimed: re-check the stored panel.
        if ctx.panel is None or ctx.request is None:
            errors = ["no validated panel was produced; call build_feature_panel"]
            report = {"ok": False, "errors": errors}
        else:
            report = validate_feature_panel(ctx.panel, ctx.request, ctx.config).model_dump()
            errors = report["errors"]
        log(f"[schema_gate] ok={not errors} {errors or ''}")
        return {"panel_report": report, "gate_errors": errors}

    def route_gate(state: PipelineState):
        if not state["gate_errors"]:
            return "transformer"
        if state["data_attempts"] >= max_data_attempts:
            raise PanelValidationError(
                f"market data failed validation after {max_data_attempts} attempts: {state['gate_errors']}")
        return "market_data"

    def transformer(state: PipelineState):
        ctx.forecasts, ctx.train_losses, ctx.last_close = run_forecasts(ctx.panel, ctx.request, ctx.config, log=log)
        write_artifacts(ctx)
        return {}

    def greeks_validation(state: PipelineState):
        msg = f"Task: {state['query']}\nValidate the {ctx.request.ticker} forecasts and price its options."
        result = greeks_agent.invoke({"messages": [{"role": "user", "content": msg}]}, run_cfg)
        log(f"[greeks_validation] {_last_text(result)}")
        if ctx.validation is None:
            raise RuntimeError("greeks/validation agent finished without calling submit_validation")
        return {"validation": ctx.validation.model_dump()}

    def final_valuation(state: PipelineState):
        table = run_valuation(ctx, ctx.validation)
        write_artifacts(ctx)
        summary = {
            "ticker": ctx.request.ticker,
            "as_of": str(ctx.panel.index[-1].date()),
            "last_close": ctx.last_close,
            "approved": ctx.validation.approved,
            "accepted_models": ctx.validation.accepted_models,
            "horizon_date": str(table["horizon_date"].iloc[0]) if len(table) else None,
            "horizon_spot": float(table["horizon_spot"].iloc[0]) if len(table) else None,
            "n_contracts": len(table),
            "issues": ctx.validation.issues,
            "notes": ctx.validation.notes,
            "output_dir": str(ctx.output_dir),
        }
        (ctx.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary}

    graph = StateGraph(PipelineState)
    for name, fn in [("market_data", market_data), ("schema_gate", schema_gate), ("transformer", transformer),
                     ("greeks_validation", greeks_validation), ("final_valuation", final_valuation)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "market_data")
    graph.add_edge("market_data", "schema_gate")
    graph.add_conditional_edges("schema_gate", route_gate, ["market_data", "transformer"])
    graph.add_edge("transformer", "greeks_validation")
    graph.add_edge("greeks_validation", "final_valuation")
    graph.add_edge("final_valuation", END)
    return graph.compile()


def run_pipeline(query, config: ForecastConfig | None = None, model=DEFAULT_MODEL, output_dir=None,
                 allow_code=True, llm=None):
    """Run the full pipeline; returns ``(final_state, ctx)``."""
    dotenv.load_dotenv(REPO_ROOT / ".env")
    output_dir = Path(output_dir or REPO_ROOT / "valuation runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    ctx = RunContext(config=config or ForecastConfig(), output_dir=output_dir)
    app = build_graph(ctx, llm or make_llm(model), allow_code=allow_code)
    return app.invoke({"query": query}), ctx


if __name__ == "__main__":
    # Agent output contains non-cp1252 characters (e.g. arrows); Windows' default console/pipe encoding can't print them.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--models", nargs="+", help="subset of transformer models to train")
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--no-code", action="store_true", help="disable the run_python tool")
    args = parser.parse_args()

    cfg = ForecastConfig(pred_len=args.pred_len, epochs=args.epochs,
                         **({"models": args.models} if args.models else {}))
    state, _ = run_pipeline(args.query, cfg, model=args.model, allow_code=not args.no_code)
    print(json.dumps(state["summary"], indent=2))
