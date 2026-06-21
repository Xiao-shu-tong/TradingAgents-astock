# TradingAgents/graph/parallel_analysts.py
"""Parallel execution primitives via ThreadPoolExecutor.

Three parallelism points:
1. Parallel Analysts — 7 analysts run concurrently (each with isolated tool loops)
2. Parallel Debate — Bull + Bear run concurrently (no tool deps)
3. Parallel Risk — Aggressive + Conservative + Neutral run concurrently (no tool deps)

Each worker gets its own LLM/state copy — no cross-branch contamination.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, ToolMessage

from tradingagents.agents import *
from tradingagents.agents.utils.agent_utils import create_msg_delete

logger = logging.getLogger(__name__)

# Map analyst type → state report key
_REPORT_KEYS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "policy": "policy_report",
    "hot_money": "hot_money_report",
    "lockup": "lockup_report",
}

# Map analyst type → creator function
_ANALYST_CREATORS = {
    "market": create_market_analyst,
    "social": create_social_media_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
    "policy": create_policy_analyst,
    "hot_money": create_hot_money_tracker,
    "lockup": create_lockup_watcher,
}


def _run_single_analyst(
    analyst_type: str,
    llm: Any,
    tool_node: Any,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Run one analyst with its tool-calling loop in isolation.

    Returns (analyst_type, partial_state_dict).
    The partial state contains the report and cleaned-up messages.
    """
    try:
        analyst_fn = _ANALYST_CREATORS[analyst_type](llm)

        # Build isolated messages: start with only the human message
        ticker = state.get("company_of_interest", "")
        isolated_state = {
            **state,
            "messages": [HumanMessage(content=ticker)],
        }

        # ── Tool-calling loop: iterate until analyst returns no tool_calls ──
        # Some analysts need multiple rounds (e.g. get_stock_data → get_indicators).
        # Max 4 rounds to prevent infinite loops with hallucinated tool calls.
        final_msgs = []
        for _round in range(4):
            result = analyst_fn(isolated_state)

            # Defensive: handle non-dict returns
            if not isinstance(result, dict):
                report_text = str(result)
                report_key = _REPORT_KEYS[analyst_type]
                return analyst_type, {report_key: report_text, "messages": []}

            if not result.get("messages"):
                break

            last_msg = result["messages"][-1] if result["messages"] else None

            # Check if the LLM wants to call tools
            if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tools_by_name = tool_node.tools_by_name
                tool_messages = []
                for tc in last_msg.tool_calls:
                    tool_fn = tools_by_name.get(tc["name"])
                    if tool_fn is None:
                        logger.warning(
                            "Unknown tool %s for %s analyst",
                            tc.get("name", "unknown"), analyst_type,
                        )
                        continue
                    try:
                        tool_output = tool_fn.invoke(tc["args"])
                        tool_messages.append(
                            ToolMessage(
                                content=str(tool_output),
                                tool_call_id=tc["id"],
                                name=tc["name"],
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "Tool %s failed for %s (round %d): %s",
                            tc["name"], analyst_type, _round, e,
                        )
                        tool_messages.append(
                            ToolMessage(
                                content=f"[Tool error: {e}]",
                                tool_call_id=tc["id"],
                                name=tc["name"],
                            )
                        )

                # Append assistant + tool messages to history, continue loop
                isolated_state["messages"] = (
                    isolated_state["messages"] + [last_msg] + tool_messages
                )
                final_msgs = result.get("messages", [])
            else:
                # No more tool calls — analyst returned final report
                final_msgs = result.get("messages", [])
                break

        # If loop exhausted without a natural conclusion, force one more
        # LLM call WITHOUT tool binding so it produces a report from what it has.
        if final_msgs:
            last_final = final_msgs[-1]
            if hasattr(last_final, "tool_calls") and last_final.tool_calls:
                logger.warning(
                    "%s analyst hit tool-loop limit (%d rounds), forcing report generation",
                    analyst_type, _round + 1,
                )
                try:
                    # Call without tools to force text generation
                    forced_result = llm.invoke(isolated_state["messages"])
                    if hasattr(forced_result, "content") and forced_result.content:
                        final_msgs = [forced_result]
                except Exception as exc:
                    logger.error("Force-report fallback also failed for %s: %s", analyst_type, exc)

        # Extract report from the final message content
        report_text = ""
        if final_msgs:
            final_msg = final_msgs[-1]
            if hasattr(final_msg, "content") and isinstance(final_msg.content, str):
                report_text = final_msg.content
            elif hasattr(final_msg, "content"):
                report_text = str(final_msg.content)

        report_key = _REPORT_KEYS[analyst_type]
        return analyst_type, {report_key: report_text, "messages": []}

    except Exception as e:
        logger.error("Analyst %s failed: %s", analyst_type, e, exc_info=True)
        report_key = _REPORT_KEYS[analyst_type]
        return analyst_type, {report_key: f"[Analyst error: {e}]", "messages": []}


def create_parallel_analyst_node(
    llm: Any,
    tool_nodes: Dict[str, Any],
    selected_analysts: List[str],
):
    """Create a LangGraph node that runs all analysts in parallel.

    Each analyst gets its own LLM, tools, and isolated message list.
    """

    def parallel_analysts_node(state: Dict[str, Any]) -> Dict[str, Any]:
        merged = {"messages": []}
        max_workers = min(len(selected_analysts), 7)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for analyst_type in selected_analysts:
                tool_node = tool_nodes.get(analyst_type)
                if tool_node is None:
                    logger.warning("No tool node for %s, skipping", analyst_type)
                    continue
                future = executor.submit(
                    _run_single_analyst, analyst_type, llm, tool_node, state,
                )
                futures[future] = analyst_type

            for future in as_completed(futures):
                analyst_type = futures[future]
                try:
                    _, partial = future.result()
                    merged.update(partial)
                except Exception as e:
                    logger.error(
                        "Analyst %s raised: %s", analyst_type, e, exc_info=True,
                    )
                    report_key = _REPORT_KEYS[analyst_type]
                    merged[report_key] = f"[Analyst crashed: {e}]"

        merged["messages"] = [HumanMessage(content="Continue")]
        return merged

    return parallel_analysts_node


# ── Parallel Debate: Bull + Bear run concurrently ──

def _run_single_debater(node_fn: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Run one debate/risk node and return its state update."""
    try:
        result = node_fn(state)
        if isinstance(result, dict):
            return result
        return {"messages": [HumanMessage(content=str(result))]}
    except Exception as e:
        logger.error("Debate/risk node failed: %s", e, exc_info=True)
        return {"messages": [HumanMessage(content=f"[Error: {e}]")]}


def create_parallel_debate_node(bull_node: Any, bear_node: Any):
    """Run Bull + Bear in parallel; merge their debate state updates."""

    def parallel_debate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"messages": []}

        with ThreadPoolExecutor(max_workers=2) as executor:
            bull_future = executor.submit(_run_single_debater, bull_node, state)
            bear_future = executor.submit(_run_single_debater, bear_node, state)

            for future in as_completed([bull_future, bear_future]):
                try:
                    partial = future.result()
                    merged["messages"].extend(partial.get("messages", []))
                    if "investment_debate_state" in partial:
                        existing = merged.get("investment_debate_state", {})
                        existing.update(partial["investment_debate_state"])
                        merged["investment_debate_state"] = existing
                except Exception as e:
                    logger.error("Debater raised: %s", e, exc_info=True)

        return merged

    return parallel_debate_node


def create_parallel_risk_node(
    aggressive_node: Any,
    conservative_node: Any,
    neutral_node: Any,
):
    """Run all 3 risk analysts in parallel; merge their state updates."""

    def parallel_risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"messages": []}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_single_debater, aggressive_node, state): "aggressive",
                executor.submit(_run_single_debater, conservative_node, state): "conservative",
                executor.submit(_run_single_debater, neutral_node, state): "neutral",
            }

            for future in as_completed(futures):
                try:
                    partial = future.result()
                    merged["messages"].extend(partial.get("messages", []))
                    if "risk_debate_state" in partial:
                        existing = merged.get("risk_debate_state", {})
                        existing.update(partial["risk_debate_state"])
                        merged["risk_debate_state"] = existing
                except Exception as e:
                    logger.error("Risk analyst raised: %s", e, exc_info=True)

        return merged

    return parallel_risk_node
