# TradingAgents/graph/setup.py

from typing import Any, Dict, List
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic
from .parallel_analysts import (
    create_parallel_analyst_node,
    create_parallel_debate_node,
    create_parallel_risk_node,
)


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals", "policy", "hot_money", "lockup"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst (technical analysis)
                - "social": Social media / sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "policy": Policy analyst (A-stock specific)
                - "hot_money": Hot money / capital flow tracker (A-stock specific)
                - "lockup": Lockup expiry / reduction watcher (A-stock specific)
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # ── Parallel Analysts: replaces sequential + Send fan-out ──
        # All 7 analysts run concurrently via ThreadPoolExecutor inside a single
        # LangGraph node. Each analyst gets its own LLM, tools, and isolated
        # message list — no cross-branch state contamination.
        parallel_analyst_node = create_parallel_analyst_node(
            llm=self.quick_thinking_llm,
            tool_nodes=self.tool_nodes,
            selected_analysts=selected_analysts,
        )

        # Create quality gate node
        quality_gate_node = create_quality_gate(self.quick_thinking_llm)

        # Create researcher and manager nodes
        # Speed optimisation: use quick_think for all nodes.
        # The 7 analysts already produce high-quality reports; deep_think on
        # synthesis nodes adds ~15-25s with marginal gain on flash-tier models.
        synthesis_llm = self.quick_thinking_llm

        # Create individual debate + risk nodes (wrapped in parallel executors)
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        parallel_debate_node = create_parallel_debate_node(
            bull_researcher_node, bear_researcher_node,
        )

        research_manager_node = create_research_manager(synthesis_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        parallel_risk_node = create_parallel_risk_node(
            aggressive_analyst, conservative_analyst, neutral_analyst,
        )

        portfolio_manager_node = create_portfolio_manager(synthesis_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes: all parallelism is INSIDE the nodes
        workflow.add_node("Parallel Analysts", parallel_analyst_node)
        workflow.add_node("Quality Gate", quality_gate_node)
        workflow.add_node("Parallel Debate", parallel_debate_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Parallel Risk", parallel_risk_node)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # ── Fully linear graph (all parallelism internal to nodes) ──
        workflow.add_edge(START, "Parallel Analysts")
        workflow.add_edge("Parallel Analysts", "Quality Gate")
        workflow.add_edge("Quality Gate", "Parallel Debate")
        workflow.add_edge("Parallel Debate", "Research Manager")
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Parallel Risk")
        workflow.add_edge("Parallel Risk", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow
