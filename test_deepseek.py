"""Test TradingAgents-Astock with DeepSeek V4"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()

# DeepSeek via Anthropic-compatible API
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "deepseek-v4-pro"
config["quick_think_llm"] = "deepseek-v4-flash"
config["backend_url"] = "https://api.deepseek.com/anthropic"

# API keys for Anthropic-compatible endpoint
os.environ["ANTHROPIC_API_KEY"] = "sk-32e46f7eb4474180b6fb3e088ed54cdf"

# A-stock data vendors (all local HTTP, no external DB needed)
config["data_vendors"] = {
    "core_stock_apis": "a_stock",
    "technical_indicators": "a_stock",
    "fundamental_data": "a_stock",
    "news_data": "a_stock",
    "signal_data": "a_stock",
}

# Debate settings
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["output_language"] = "Chinese"

print("=" * 60)
print("TradingAgents-Astock + DeepSeek V4 Test")
print("Ticker: 688256 (寒武纪)")
print("Trade date: 2026-06-18")
print("LLM: deepseek-v4-pro/flash via Anthropic API")
print("=" * 60)

ta = TradingAgentsGraph(debug=False, config=config)

try:
    _, decision = ta.propagate("688256", "2026-06-18")
    print("\n" + "=" * 60)
    print("FINAL DECISION:")
    print("=" * 60)
    print(decision)
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
