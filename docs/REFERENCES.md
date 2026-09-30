# References

Prior work supporting the design of the Smart Guided Troubleshooting Engine (SGTE), grouped by the component it
informs.

## Retrieval-augmented generation and grounding

| Reference | Relevance to SGTE |
|---|---|
| Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | Foundational RAG work; basis for grounding plans in SIIS text. |
| Yan, S.-Q. et al. (2024). *Corrective Retrieval Augmented Generation.* [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) | Evaluates retrieval quality and falls back when it is poor; supports the cross-encoder gate and `no_match` fallback. |
| Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys. [arXiv:2202.03629](https://arxiv.org/abs/2202.03629) | Motivates copying steps from source sentences and keeping URIs out of LLM schemas. |
| Gao, T. et al. (2023). *Enabling Large Language Models to Generate Text with Citations.* EMNLP. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) | Supports attributing every output step to its source. |

## Hybrid retrieval, fusion and reranking

| Reference | Relevance to SGTE |
|---|---|
| Robertson, S. and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval 3(4). | Lexical retrieval (BM25) in SIIS RAG and the deeplink engine. |
| Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) | Dense retrieval component. |
| Cormack, G. V., Clarke, C. L. A. and Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR. | Reciprocal rank fusion (RRF) of BM25 and dense rankings in the deeplink engine. |
| Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) | Cross-encoder reranking and gating. |
| Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) | Shows BM25 + reranking is robust out of domain; supports the hybrid design on an unseen kit. |

## Models

| Reference | Relevance to SGTE |
|---|---|
| Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) | Framework behind all-MiniLM-L6-v2. |
| Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957) | Base architecture of the embedding and reranking models. |
| Nguyen, T. et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.* [arXiv:1611.09268](https://arxiv.org/abs/1611.09268) | Training data of ms-marco-MiniLM-L-6-v2. |
| Xiao, S. et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding.* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) | BGE embedding family (bge-small-en-v1.5, cache embedder). |
| Qwen Team (2024). *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) | Local SLM (Qwen2.5-1.5B-Instruct). |

## Semantic cache and vector search

| Reference | Relevance to SGTE |
|---|---|
| Johnson, J., Douze, M. and Jégou, H. (2019). *Billion-scale Similarity Search with GPUs.* IEEE Transactions on Big Data. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734) | FAISS index behind the semantic cache. |
| Douze, M. et al. (2024). *The Faiss Library.* [arXiv:2401.08281](https://arxiv.org/abs/2401.08281) | Current FAISS reference. |
| Bang, F. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings.* NLP-OSS Workshop. | Closest prior work to the tiered cache and threshold τ. |

## Structured output

| Reference | Relevance to SGTE |
|---|---|
| Willard, B. T. and Louf, R. (2023). *Efficient Guided Generation for Large Language Models.* [arXiv:2307.09702](https://arxiv.org/abs/2307.09702) | JSON-constrained LLM output matching the `schema.py` contract. |

## Troubleshooting and case reuse

| Reference | Relevance to SGTE |
|---|---|
| Heckerman, D., Breese, J. S. and Rommelse, K. (1995). *Decision-Theoretic Troubleshooting.* Communications of the ACM 38(3). | Classic basis for ordering troubleshooting steps. |
| Aamodt, A. and Plaza, E. (1994). *Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches.* AI Communications 7(1). | Frames reuse of cached plans for similar complaints. |

## Evaluation

| Reference | Relevance to SGTE |
|---|---|
| Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | Standard RAG faithfulness/relevance metrics to compare against `metrics.md`. |
