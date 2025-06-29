# 📄 PaperTalks

**PaperTalks** is a lightweight, locally run application that allows users to interact with any PDF or text document. It supports two core functionalities:
1. Generate **Multiple Choice Questions (MCQs)** using classical NLP.
2. Ask **context-aware questions** to the document using a custom **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Features

- 📚 Upload any PDF or `.txt` file.
- 🧠 Generate MCQs using spaCy (no LLMs involved).
- 💬 Chat with the document using LangChain-based RAG pipeline.
- 🧾 Runs fully locally — uses **FAISS** for retrieval and **TinyLlama** for generation.
- 🖥️ Clean, interactive UI built with **Streamlit**.

---

## 🛠️ Tech Stack

- **Python**
- **LangChain** – for building modular retrieval and generation chains.
- **FAISS** – for fast vector similarity search.
- **TinyLlama (1.1B)** – as a local LLM for context-based generation.
- **spaCy** – for traditional NLP tasks and MCQ generation.
- **Streamlit** – for the front-end interface.

---

## ⚙️ How It Works

### 🧠 MCQ Generator
- Utilizes spaCy to perform named entity recognition (NER) and pattern matching.
- Applies rule-based logic to formulate question-answer pairs.
- Designed for speed and runs without any LLM or external API.

### 💬 RAG-based Document Chat
- Documents are chunked and embedded using LangChain’s document loaders and embedding models.
- Vector representations are stored in a local FAISS index.
- On user query:
  - Top-k relevant chunks are retrieved using cosine similarity.
  - A prompt is dynamically generated using the query and retrieved context.
  - **TinyLlama**, running locally, generates the final response.
- Includes fallback handling for low-relevance cases.

---

## 🧪 Running Locally

