from __future__ import annotations

import os

from research_assistant.config import RuntimeConfig
from research_assistant.graph import ResearchAssistant


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run provider-backed example.")
        return
    print(
        "Provider-backed adapter example is intentionally left as an integration point. "
        "Inject retriever/generator adapters that satisfy the project protocols."
    )
    _ = RuntimeConfig.from_env()
    _ = ResearchAssistant


if __name__ == "__main__":
    main()
