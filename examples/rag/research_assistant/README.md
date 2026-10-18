# Research Assistant (Self-RAG evolution)

This package provides a modular, testable first release of an original AI research assistant built alongside the archival notebook at `/examples/rag/langgraph_self_rag.ipynb`.

## Purpose

- Preserve the original notebook unchanged for historical reference.
- Add a maintainable implementation with bounded retries, citation-aware claims, and deterministic offline tests.
- Support future API and dashboard work through clear interfaces.

## Architecture

- `src/research_assistant/config.py`: runtime limits from environment variables.
- `src/research_assistant/models.py`: report, evidence, citation, status, and event models.
- `src/research_assistant/retrieval.py`: retriever and query-rewriter protocols, stable evidence IDs.
- `src/research_assistant/generation.py`: structured report parsing, citation checks, unsupported-claim checks.
- `src/research_assistant/graph.py`: bounded retrieval, rewrite, generation, correction, and terminal outcomes.
- `src/research_assistant/observability.py`: structured events plus usage/latency collection when metadata exists.
- `src/research_assistant/evaluation.py`: deterministic baseline-vs-improved comparison helpers.

## Terminal statuses

- `answered`
- `insufficient_evidence`
- `citation_validation_failed`
- `model_error`
- `retry_limit_exceeded`

## Setup

```bash
cd /home/runner/work/langgraph-self-rag-assistant/langgraph-self-rag-assistant/examples/rag/research_assistant
make install
```

## Environment variables

- `RESEARCH_ASSISTANT_MAX_RETRIEVAL_ATTEMPTS`
- `RESEARCH_ASSISTANT_MAX_CORRECTION_ATTEMPTS`
- `RESEARCH_ASSISTANT_MAX_GRAPH_STEPS`
- `RESEARCH_ASSISTANT_MIN_EVIDENCE_CHUNKS`
- `RESEARCH_ASSISTANT_MAX_RETRIEVED_CHUNKS`

Provider-backed integrations should read credentials from environment variables only (for example, optional `OPENAI_API_KEY` in a custom adapter), and secrets must never be committed.

## Usage

```python
from research_assistant import ResearchAssistant, RuntimeConfig
```

Inject retriever/generator implementations that satisfy the declared protocols and call `run(question)`.

## Testing and quality checks

```bash
make format
make lint
make test
```

The included tests are deterministic and offline.

## Evaluation example

Run a reproducible deterministic comparison:

```bash
uv run python examples/compare_workflows.py
```

The printed result is measured from local fake components. Any provider-backed run is illustrative and depends on external models and credentials.

## Error handling included

- Empty retrieval results.
- Retriever exceptions.
- Generator exceptions.
- Malformed structured outputs.
- Invalid citation IDs.
- Unsupported claims.
- Exhausted correction/retrieval retries.

## Limitations

- Claim support checking is lexical and intentionally simple for deterministic testing.
- Provider adapters are intentionally out-of-scope for this first release and should be injected through protocols.

## Attribution

This work is inspired by the archival LangGraph Self-RAG notebook and the Self-RAG research direction. The original notebook remains unchanged and available at `/examples/rag/langgraph_self_rag.ipynb`.
