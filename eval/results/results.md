# NewsLens evaluation

Sample corpus: 22 articles, 9 labelled stories (fictional). Analyst/writer: offline mock.

## 1. Story clustering

| distance threshold | stories | ARI | homogeneity | completeness |
|--:|--:|--:|--:|--:|
| 0.5 | 17 | 0.458 | 1.000 | 0.760 |
| 0.6 | 13 | 0.723 | 1.000 | 0.854 |
| 0.7 | 9 | 1.000 | 1.000 | 1.000 |
| 0.8 **(default)** | 9 | 1.000 | 1.000 | 1.000 |
| 0.9 | 9 | 1.000 | 1.000 | 1.000 |

## 2. Fabricated-claim detection

14 paraphrased true claims, 34 fabricated variants.

| fabrication type | caught |
|:--|--:|
| altered_number | 10/10 |
| swapped_entity | 10/10 |
| invented_fact | 14/14 |
| **all** | **34/34** |

- True claims accepted: **14/14** (false-positive rate 0%)
- Precision 1.00 · Recall 1.00 · F1 1.00 (missed fabrications: 0)

## 3. End-to-end injection test

Analyst injected 9 fabricated claims (one per story).

- Marked unsupported by the verifier: **9**
- Fabrications that reached the published briefing: **0/9**
- Bullets published: 9, every one with ≥1 source citation; writer revisions: 1
- Stories covered by 2+ publishers: 6/9
