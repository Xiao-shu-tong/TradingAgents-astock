"""Verify structured output fix for DeepSeek V4 thinking mode"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["ANTHROPIC_API_KEY"] = "sk-32e46f7eb4474180b6fb3e088ed54cdf"

# Test 1: Anthropic path - with_structured_output should disable thinking
print("=" * 60)
print("Test 1: Anthropic path - structured output with DeepSeek V4")
print("=" * 60)

from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic

llm = NormalizedChatAnthropic(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com/anthropic",
    api_key="sk-32e46f7eb4474180b6fb3e088ed54cdf",
)

# Check that thinking is not set (DeepSeek defaults to on)
print(f"thinking before structured_output: {getattr(llm, 'thinking', 'not set')}")

from pydantic import BaseModel

class TestSchema(BaseModel):
    rating: str
    reason: str

# This should NOT raise - should disable thinking and work
try:
    structured_llm = llm.with_structured_output(TestSchema)
    print(f"structured_llm created: {type(structured_llm).__name__}")
    # Check that thinking was restored
    print(f"thinking after structured_output: {getattr(llm, 'thinking', 'not set')}")
    print("PASS: Structured output created without error")
except Exception as e:
    print(f"FAIL: {e}")

print()

# Test 2: OpenAIClient path - DeepSeekChatOpenAI should raise NotImplementedError
print("=" * 60)
print("Test 2: OpenAI path - DeepSeekChatOpenAI V4 detection")
print("=" * 60)

from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

llm2 = DeepSeekChatOpenAI(
    model="deepseek-v4-flash",
    api_key="sk-32e46f7eb4474180b6fb3e088ed54cdf",
)

try:
    llm2.with_structured_output(TestSchema)
    print("FAIL: Should have raised NotImplementedError")
except NotImplementedError as e:
    print(f"PASS: Correctly raised NotImplementedError for deepseek-v4-flash")
except Exception as e:
    print(f"UNEXPECTED: {e}")

print()
print("All tests done!")
