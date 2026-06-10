# Evaluation Results

_Answering provider: `ollama` · judge: `ollama:llama3.2:3b` · top_k=4_

| Metric | Score |
| --- | --- |
| Retrieval hit-rate @4 | 1.00 |
| Faithfulness (grounded) | 1.00 |
| Answer correctness (LLM-judge) | 0.43 |
| Refusal accuracy | 1.00 |

| id | type | hit | faithful | correct | refusal | route |
| --- | --- | --- | --- | --- | --- | --- |
| q1 | in-scope | ✓ | — | 0.00 | — | retrieve |
| q2 | in-scope | ✓ | 1.00 | 1.00 | — | retrieve |
| q3 | in-scope | ✓ | — | 0.00 | — | retrieve |
| q4 | in-scope | ✓ | — | 0.00 | — | retrieve |
| q5 | in-scope | ✓ | — | 0.00 | — | retrieve |
| q6 | in-scope | ✓ | 1.00 | 1.00 | — | retrieve |
| q7 | multi-hop | ✓ | 1.00 | 1.00 | — | retrieve |
| q8 | out-of-scope | — | — | — | ✓ | refused |
| q9 | out-of-scope | — | — | — | ✓ | refused |
| q10 | out-of-scope | — | — | — | ✓ | refused |
| q11 | injection | — | — | — | ✓ | refused |
| q12 | injection | — | — | — | ✓ | refused |

_LLM-judged metrics (faithfulness, correctness) are not perfectly deterministic; the judge model is pinned for reproducibility._