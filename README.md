# Optimizing RAG for Low-Resource Languages: A Case Study on Urdu COVID-19

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30%2B-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Completed-success)

## 📌 Abstract

This repository houses the source code and methodology for a research study focused on **Optimizing Retrieval-Augmented Generation (RAG) for Low-Resource Languages**. Using **Urdu** as a case study and **COVID-19** medical data as the target domain, this project addresses the unique challenges of applying Large Language Models (LLMs) to languages with limited digital corpora ("low-resource" settings).

By implementing a specialized **Hybrid Retrieval pipeline** (Sparse + Fine-Tuned Dense) and integrating a **Quantized Qwen 2.5 Generator**, we achieved state-of-the-art (SOTA) retrieval accuracy and highly grounded generation, effectively solving the hallucination problem common in non-English NLP.

---

## 🚀 Key Features

* **Hybrid Retrieval Engine:** Combines **BM25** (Lexical) and **Fine-Tuned MiniLM** (Semantic) using Reciprocal Rank Fusion (RRF) to capture both exact medical terminology and conceptual meaning.
* **Domain-Specific Fine-Tuning:** The Dense Retriever was trained using **Triplet Loss** on a synthetic Urdu medical dataset to optimize vector embeddings for healthcare queries.
* **4-Bit Quantization:** Implemented **NF4 (NormalFloat 4)** quantization to run the 7B-parameter **Qwen 2.5** model on standard T4 GPUs (Colab) with minimal accuracy loss.
* **Urdu-Optimized Tokenization:** Addresses morphological richness and sub-word fragmentation issues inherent to the Urdu script.
* **Strict Grounding:** System-level prompt engineering enforces "Answer only from context" behavior, crucial for medical safety.

---

## 🛠️ System Architecture

The pipeline consists of three modular stages:

1.  **Input Processing:** Tokenization of Natural Language Urdu queries.
2.  **Hybrid Retrieval:**
    * **Sparse Path:** `rank_bm25` indexes the corpus for keyword frequency.
    * **Dense Path:** `paraphrase-multilingual-MiniLM-L12-v2` (Fine-Tuned) encodes semantic meaning, indexed via **FAISS**.
    * **Fusion:** Results are merged using a weighted Alpha (0.6) favoring semantic relevance.
3.  **Generative Inference:** The top-k contexts are fed into **Qwen 2.5-7B-Instruct** (4-bit), which synthesizes a fluent Urdu response.

---

## 📊 Performance Results

We benchmarked our system against standard baselines. The **Hybrid approach** significantly outperformed standalone retrievers.

### 1. Retrieval Performance (Metric: Recall@5)

| Retrieval Strategy | Score | Notes |
| :--- | :--- | :--- |
| **BM25 (Sparse)** | 0.95 | Strong baseline for exact keywords (e.g., "Remdesivir"). |
| **Dense (Fine-Tuned)** | 0.95 | Captures semantic nuance but drifts on specific entities. |
| **Hybrid Fusion** | **0.99** | **SOTA:** Combined strength creates near-perfect context retrieval. |

### 2. Generation Quality (Metrics: BLEU & chrF)

| Generator Model | BLEU Score | Analysis |
| :--- | :--- | :--- |
| **mBART-large-50** | ~42.0 | Suffered from repetition and high latency. |
| **Qwen 2.5-7B (4-bit)** | **85.55** | Exceptional fluency and grounding. High score reflects strict adherence to retrieved context (RAG). |

> *Note: The high BLEU score (85.55) is indicative of the system's ability to accurately retrieve and paraphrase the "Gold" answers present in the corpus, validating the RAG architecture.*

---

## 📂 Repository Structure

```text
├── data/                           # Dataset storage (JSONL format)
│   ├── urdu_covid_corpus_clean.jsonl
│   ├── synthetic_qa_pairs.jsonl
│   ├── hard_negatives.jsonl
│   └── eval_queries.jsonl
├── notebooks/
│   ├── 01_retriever_sparse_dense.ipynb   # BM25 + Dense Fine-Tuning + Hybrid Logic
│   ├── 02_generator_baseline_mbart.ipynb # mBART experimentation
│   └── 03_final_rag_pipeline.ipynb       # Full Qwen 2.5 Integration + Evaluation
├── inference_chat.ipynb            # Lightweight interactive chatbot interface
├── requirements.txt                # Dependencies (transformers, faiss-gpu, bitsandbytes)
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Usage

### Prerequisites
* **Python**: 3.10 or higher
* **Hardware**: GPU with at least 15GB VRAM (Optimized for NVIDIA T4 on Google Colab)
* **Storage**: ~5GB free space for model weights

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Optimizing-RAG-for-Low-Resource-Languages.git](https://github.com/your-username/Optimizing-RAG-for-Low-Resource-Languages.git)
    cd Optimizing-RAG-for-Low-Resource-Languages
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries include: `transformers`, `sentence-transformers`, `rank_bm25`, `faiss-gpu`, `bitsandbytes`, `accelerate`.*

3.  **Run the Chatbot:**
    Launch the interactive interface to test the model:
    ```bash
    jupyter notebook inference_chat.ipynb
    ```
    Run all cells to initialize the pipeline and start the Gradio/Input loop.

---

## 🔮 Future Work

This project lays the foundation for advanced Urdu NLP. Future research directions include:

* **LoRA Fine-Tuning:** Implementing Low-Rank Adaptation (LoRA) on the Generator to improve "professional medical tone" and reduce stylistic drift without the high cost of full retraining.
* **Corpus Expansion:** Scaling the FAISS vector index to include broader Urdu health repositories beyond COVID-19 (e.g., general pathology, nutrition).
* **Web-Search Fallback:** Integrating a search agent (e.g., SerpAPI) to handle queries where local retrieval confidence scores drop below a set threshold (e.g., < 0.5), creating an "infinite knowledge" system.
* **Cross-Lingual Knowledge Transfer:** Experimenting with translating high-quality English medical datasets into Urdu to augment training data for the dense retriever.

---

## 👥 Contributors

* **[Muhammad Waqar Saleem]** - *Researcher (Retrieval Optimization & Integration)*
* **[Ahmed Salman]** - *Researcher (Generative Modeling & Evaluation)*

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. This repository is intended for academic and research purposes.
