# Autorship-Attribution-Computational-Linguistics

**Computational Linguistics II Project – A.Y. 2024/2025** **Author:** Matilde Viti

### [Read the full report](https://github.com/matildeviti22/Autorship-Attribution-Computational-Linguistics/blob/main/RELAZIONE_VITI_LC2%20copia.pdf)

## Project Overview
This project addresses the task of **Authorship Attribution**: identifying the author of a text based on their unique linguistic "fingerprint." The study focuses on classifying literary paragraphs from three iconic English novelists: **Charlotte Brontë**, **George Eliot**, and **Jane Austen**.

The core objective was to compare how different text representations—ranging from traditional stylometric features to state-of-the-art Transformers—impact the model's ability to distinguish between authorial styles.

## Pipeline
The project follows a rigorous computational pipeline:

1.  **Data Engineering:** A custom preprocessing pipeline to clean raw Project Gutenberg texts (metadata removal, narrative filtering), segmenting them into paragraphs (50-100 tokens), and balancing classes via undersampling.
2.  **Linguistic Analysis:** Extraction of non-lexical morphosyntactic and structural features using the **Profiling-UD** system.
3.  **Modeling & Evaluation:**
    * **Stylometric SVM:** Classification based on structural features (lexical density, punctuation distribution, syntactic complexity).
    * **Multi-modal N-gram SVM:** Comparison of various n-gram configurations (words, lemmas, characters, and POS).
    * **Word Embeddings:** Exploration of different aggregation strategies (mean, sum, max) using pre-trained **UKWAC** embeddings.
    * **Deep Learning:** Fine-tuning **DistilBERT-base-cased**, featuring training/validation loss analysis and optimal checkpoint selection.

## Key Findings
* **Model Performance:** Both **DistilBERT** and the **N-gram SVM** achieved nearly identical top results (Accuracy ~83-84%), proving that local linguistic patterns remain highly competitive for stylistic tasks.
* **Feature Importance:** Stylometric analysis revealed that punctuation usage and lexical density are among the strongest discriminators for identifying these authors.

## Tech Stack
* **Language:** Python
* **NLP:** Hugging Face Transformers, Profiling-UD, Spacy
* **Machine Learning:** Scikit-learn (SVM, Preprocessing, Evaluation)
* **Deep Learning:** PyTorch
* **Data Handling:** Pandas, JSON, CoNLL-U format
* **Infrastructure:** Google Colab (NVIDIA T4 GPU)

---

### Final Thoughts
This project highlights that Authorship Attribution is not just about model complexity, but about the **richness of text representation**. It demonstrates that even simpler models can be highly effective when they successfully capture the underlying syntactic and structural habits of an author.
