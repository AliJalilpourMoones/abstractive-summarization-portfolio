# Abstractive Text Summarization with BART

This repository contains the source code for a fine-tuned abstractive text summarization model. The project leverages the pre-trained **BART (Bidirectional Auto-Regressive Transformer)** model from Hugging Face and fine-tunes it on the **XSum (Extreme Summarization)** dataset to generate concise, single-sentence summaries from news articles.



---

## Abstract
This project demonstrates a complete workflow for fine-tuning a large-scale, pre-trained sequence-to-sequence model for a generative NLP task. By using the Hugging Face `Trainer` API, this repository provides an efficient and professional implementation of an abstractive summarization pipeline. The model's performance is evaluated using the standard ROUGE metric, and a qualitative analysis of the generated summaries is presented. This work showcases key skills in generative AI, model evaluation, and modern NLP tooling.

---

## 1. Model Architecture: BART

This project uses **BART (`facebook/bart-large-cnn`)**, a powerful sequence-to-sequence model ideal for summarization. Unlike encoder-only models like BERT, BART has both:
* An **Encoder** that reads the source article and creates a rich numerical representation of its meaning.
* A **Decoder** that uses this representation to generate a completely new summary word by word.

This architecture allows the model to be "abstractive"—writing novel sentences rather than just copying and pasting existing ones.

---

## 2. Dataset: XSum

The model is fine-tuned on the **XSum (Extreme Summarization)** dataset. This dataset is particularly challenging and well-suited for this task because it consists of BBC news articles paired with professionally written, single-sentence summaries that are highly abstractive. The data is loaded and processed efficiently using the Hugging Face `datasets` library.

---

## 3. Results & Analysis

**[RESULTS PENDING]**
The model will be trained for 3 epochs. The primary evaluation metrics are the ROUGE scores, which measure the overlap between the generated summaries and the human-written reference summaries.

| Metric  | Score                |
| :------ | :------------------- |
| ROUGE-1 |                      |
| ROUGE-2 |                      |
| ROUGE-L |                      |

### ### Qualitative Analysis

* **Success Cases:** The model is expected to generate fluent, grammatically correct summaries that capture the main entity and action of the source article.
    * **Failure Cases (Error Analysis):** The model may sometimes struggle with highly technical or jargon-heavy articles. It might also occasionally "hallucinate" or generate facts not present in the source text, which is a common challenge for generative models.
    ---

## 4. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AliJalilpourMoones/abstractive-summarization-portfolio.git](https://github.com/AliJalilpourMoones/abstractive-summarization-portfolio.git)
    cd abstractive-summarization-portfolio
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the training script:**
    ```bash
    python src/train.py
    ```
    The script will download the model and dataset, then begin training. Results and saved models will be placed in the `/results` directory.