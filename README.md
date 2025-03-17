# 📝 PDF QnA Helper – Local RAG Bot  

## 🚀 Overview  
This is a **lightweight, privacy-focused RAG (Retrieval-Augmented Generation) bot** designed to **help with research papers,
coursework, and large PDFs**—without relying on external APIs.  

### **Why?**  
As a bioinformatics graduate, reading **dense research papers** was frustrating, especially with large, complex PDFs. **LLMs can help**, 
but most require **cloud processing, raising privacy concerns** for copyrighted coursework. This project solves that by keeping everything 
**local** while ensuring efficient QnA responses.

> 💡 **Note:** This project requires **at least 8GB of RAM** to run smoothly. Lower memory may lead to performance issues.


## 🏗️ Reasoning  
| Component            | Model/Tool Used                 | Why? |
|----------------------|--------------------------------|------|
| **LLM**             | `mistral-7b-v3` (via Ollama)   | Open-source, fast, and outperforms larger models in efficiency |
| **Embeddings**      | `all-mpnet-base-v2`            | Lightweight and powerful for sentence embeddings |
| **Vector DB**       | FAISS                          | Efficient similarity search |
| **Retrieval**       | Cosine Similarity Search      | Finds the most relevant PDF chunks |
| **Processing**      | PyMuPDF / pdfplumber           | Handles complex PDFs |

---

## 📂 Directory Structure  
```bash
├──    data/                 # Stores PDFs (e.g., “biochem.pdf”)
├──    src/                  # Main application source code
│   ├── main.py             # Entry point for the app
│   ├── models.py           # Manages LLM and embedding model
│   ├── pdf_processing.py   # Handles PDF text extraction & chunking
│   ├── query_script.py     # Handles user queries (RAG pipeline)
├── .gitignore              # Ignores cache, pycache, and other unnecessary files
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
```

## 🔧 Installation & Setup 

1️⃣ **Clone the repo**  
```bash
git clone hhttps://github.com/A1pha-Z3r0/BioRAG.git
cd BioRAG
```
2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```
3️⃣ **Download models from Ollama**
```bash
ollama pull mistral
```
4️⃣ **Run the src/main.py script**
```bash
python src/pdf_processing.py --pdf data/biochem.pdf
```

5️⃣**Ask Question**
```Bash
Enter your question: _________
```

## 💾 System Requirements

To run this project efficiently, your system should have:
	•	✅ At least 8GB of RAM (recommended for smooth execution)
	•	✅ A CPU with AVX support (for optimal FAISS performance)
	•	✅ Ollama installed to run Mistral locally

## 📌 Features

✅ Fully Local RAG QnA – No cloud dependency
✅ Handles Large PDFs – Efficient text chunking & retrieval
✅ Lightweight Embeddings – Keeps resource usage low
✅ Fast Inference – Uses Mistral 7B for speed and efficiency
✅ Privacy-Focused – No data leaves your machine