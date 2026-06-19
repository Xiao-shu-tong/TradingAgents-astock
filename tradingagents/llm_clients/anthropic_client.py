from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
    "thinking",
)


# DeepSeek model prefixes that enable thinking mode by default
_DEEPSEEK_THINKING_PREFIXES = ("deepseek-v4", "deepseek-reasoner", "deepseek-r1")


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.

    DeepSeek V4 models routed through the Anthropic-compatible endpoint
    automatically enable thinking mode, which does not support tool_choice.
    ``with_structured_output`` temporarily disables thinking to make
    structured output (function-calling) work, then restores it.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        # DeepSeek thinking models don't support tool_choice.
        # Temporarily disable thinking so structured output works.
        model = getattr(self, "model", "") or ""
        if model.startswith(_DEEPSEEK_THINKING_PREFIXES):
            saved_thinking = getattr(self, "thinking", None)
            self.thinking = {"type": "disabled"}
            try:
                return super().with_structured_output(schema, method=method, **kwargs)
            finally:
                self.thinking = saved_thinking
        return super().with_structured_output(schema, method=method, **kwargs)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
