from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for monitoring agent actions and observations."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"Prompt to LLM was:\n {prompts[0]}")
        print("-------------------------")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """RUn when LLM ends running."""
        print(f"LLM response: {response.generations[0][0].text}")
        print("-------------------------")
