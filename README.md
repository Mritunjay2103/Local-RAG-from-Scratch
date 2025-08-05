# 🧠 Local Retrieval-Augmented Generation (RAG) with PDFs & Open-Source Models

This project walks you through building a **local Retrieval-Augmented Generation (RAG)** pipeline from scratch, running entirely on your own machine — no external APIs required.  
We’ll use **PDF ingestion → text chunking → embeddings → semantic search → local LLM generation** to create a complete end-to-end application.  

---

<img width="1225" height="678" alt="image" src="https://github.com/user-attachments/assets/c210a56c-2123-4899-8e2f-4d95bf83db76" />


## 📚 Table of Contents
1. [Overview](#overview)  
2. [Why RAG & Why Local?](#why-rag--why-local)  
3. [What We’re Building](#what-were-building)  
4. [Implementation Steps](#implementation-steps)  
5. [Technologies Used](#technologies-used)  
6. [Potential Extensions](#potential-extensions)  
7. [Resources](#resources)  

---

## 📝 Overview
This project is inspired by the **NVIDIA GTC session** on building local RAG pipelines.  
It covers:
- Loading and processing PDF documents.
- Preprocessing into **semantic chunks**.
- Creating **vector embeddings** using open-source models.
- Building a **semantic search pipeline** for retrieval.
- Running a **local LLM** to answer questions augmented with retrieved context.

---

## ❓ Why RAG & Why Local?

### Why RAG?
- Enhances LLM responses with **factual grounding**.
- Reduces hallucinations by providing **retrieved context**.
- Enables **domain-specific QA**.

### Why Run Locally?
- **Data privacy** — no cloud API calls.
- **Full control** over performance and cost.
- Works **offline**.
- Uses **open-source models** with GPU acceleration.

---

## 🚀 What We’re Building
At a high level, we will:
1. **Import & process a PDF** for text extraction.
2. **Split the text** into semantically meaningful chunks.
3. **Generate embeddings** from the chunks.
4. **Index embeddings** for fast semantic retrieval.
5. **Retrieve relevant chunks** based on a query.
6. **Augment a local LLM’s prompt** with retrieved context.
7. **Generate a grounded, factual answer**.

---

## 🛠 Implementation Steps

### **Part 0 – Resources & Overview**
- Introduction to RAG and local deployment.
- Key papers, repos, and tools.

### **Part 1 – Understanding RAG**
- What is RAG?  
- Why it’s important for modern AI workflows.  
- Practical applications (QA, summarization, search, chatbots).

### **Part 2 – Setting Up**
- Project structure.  
- Dependencies installation.  

### **Part 3 – PDF Import & Processing**
- Using `pypdf` / `pdfplumber` to load PDF content.  
- Cleaning extracted text.

### **Part 4 – Text Preprocessing & Chunking**
- Sentence splitting.
- Creating overlapping chunks for better retrieval.

### **Part 5 – Embeddings**
- Choosing an **open-source embedding model**.  
- Running embeddings on CPU vs GPU.  
- Vector store setup (e.g., FAISS).

### **Part 6 – Retrieval**
- Implementing **semantic search**.
- Cosine similarity & vector distance metrics.

### **Part 7 – Similarity Measures**
- Euclidean, cosine similarity, dot product.

### **Part 8 – Retrieval Functions**
- Modularizing semantic search.

### **Part 9 – Local LLM Setup**
- Selecting an appropriate LLM.  
- Running with libraries like `llama.cpp`, `GPT4All`, or `transformers`.

### **Part 10 – Generation**
- Passing user query + retrieved context into the LLM.  
- Formatting prompts for better results.

### **Part 11 – Context Augmentation**
- Prompt engineering with context injection.

### **Part 12 – Putting It All Together**
- Unified RAG pipeline: `query → retrieve → augment → generate`.

---

## 💻 Technologies Used
- **Python 3.11+**
- **PyPDF2 / pdfplumber** – PDF parsing.
- **NLTK / spaCy** – Sentence segmentation.
- **SentenceTransformers** – Embedding models.
- **FAISS / ChromaDB** – Vector search.
- **Transformers** – LLM inference.
- **CUDA / PyTorch** – GPU acceleration.

---

## 🔮 Potential Extensions
- Multi-document ingestion.
- Support for images & OCR in PDFs.
- Streaming responses from LLM.
- Hybrid search (keyword + semantic).
- Evaluation metrics for retrieval quality.

---

## 📌 Resources
- 📄 [Original RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)  
- 🖥 [Sentence Transformers Documentation](https://www.sbert.net/)  
- ⚡ [FAISS Documentation](https://faiss.ai/)  
- 🐍 [Transformers by Hugging Face](https://huggingface.co/docs/transformers/index)  
