# Agentic RAG Chatbot for Multi-Format Document QA using Model Context Protocol (MCP)

This project implements an intelligent Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions based on information extracted from various document formats. It features an agent-based architecture with inter-agent communication facilitated by a custom Model Context Protocol (MCP).

## ✨ Features

- **Multi-Format Document Support**: Upload and process documents in PDF, CSV, DOCX, PPTX, TXT, and Markdown formats.
- **Agentic Architecture**: A modular design with specialized agents for document ingestion, information retrieval, and LLM-based response generation.
- **Model Context Protocol (MCP)**: Agents communicate using structured messages defined by the MCP, ensuring clear data flow and traceability.
- **Semantic Search**: Utilizes FAISS and Sentence Transformers for efficient vector-based similarity search to retrieve relevant document chunks.
- **Local LLM Integration**: Leverages Ollama to run large language models (like Llama3) locally for privacy and cost-efficiency.
- **Interactive Chatbot UI**: A user-friendly interface built with Streamlit for seamless interaction.

## 🧠 Architecture Overview

The system is built around a `CoordinatorAgent` that orchestrates the workflow between three core agents:

### 1. IngestionAgent

- **Role**: Responsible for parsing uploaded documents and extracting their text content.
- **Input**: `DOCUMENT_UPLOAD` message from the UI (via Coordinator).
- **Output**: `DOCUMENT_PARSED` message containing document ID, filename, and extracted text.

### 2. RetrievalAgent

- **Role**: Manages the vector store, indexes document chunks, and retrieves relevant context for user queries.
- **Input**: `DOCUMENT_PARSED` message (for indexing) or `QUERY_REQUEST` message (for retrieval).
- **Output**: `DOCUMENT_INDEXED` (confirmation) or `CONTEXT_RESPONSE` (containing retrieved text chunks and query).

### 3. LLMResponseAgent

- **Role**: Crafts a prompt using the retrieved context and the user's query, then calls the local LLM (Ollama) to generate an answer.
- **Input**: `CONTEXT_RESPONSE` message from the RetrievalAgent.
- **Output**: `ANSWER_RESPONSE` message containing the final answer, and a structured `RETRIEVAL_RESULT` (MCP format) for transparency.

## 🔄 System Flow Diagram (with Message Passing)

```mermaid
graph TD
    A[User Interface (Streamlit)] -->|Document Upload| B(CoordinatorAgent);
    B -->|DOCUMENT_UPLOAD (filename, content)| C(IngestionAgent);
    C -->|DOCUMENT_PARSED (doc_id, content, filename)| D(RetrievalAgent);
    D -->|DOCUMENT_INDEXED (doc_id, filename)| B;

    A -->|User Query| B;
    B -->|QUERY_REQUEST (query)| D;
    D -->|CONTEXT_RESPONSE (retrieved_context, query, full_context_details)| E(LLMResponseAgent);
    E -->|ANSWER_RESPONSE (answer, retrieval_mcp_output, original_sources_info)| B;
    B -->|Answer + Sources| A;
```

> **Note**: Error messages and direct ERROR message types are omitted from the main flow for simplicity but are handled by agents.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Agent Orchestration**: Custom Python classes with Pydantic for MCP message validation  
- **Document Parsing**: PyPDF2, pandas, python-docx, python-pptx  
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)  
- **Vector Store**: FAISS (Facebook AI Similarity Search)  
- **LLM Integration**: Ollama Python client  
- **Text Processing**: nltk (for sentence tokenization)  
- **Logging**: Python's built-in logging module  

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+**  
- **Ollama**: Download from [https://ollama.com](https://ollama.com)

### Install Llama3 Model

```bash
ollama pull llama3
```

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Create `requirements.txt`

```txt
streamlit
PyPDF2
pandas
python-docx
python-pptx
sentence-transformers
faiss-cpu
numpy
ollama
pydantic
nltk
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 📁 Project Structure

```
your_rag_chatbot_project/
├── agents.py             # Contains all agent logic, MCP, VectorStore, parsers
├── app.py                # Streamlit UI for the chatbot
├── requirements.txt      # List of Python dependencies
```

---

## ▶️ Running the Application

### Step 1: Ensure Ollama is Running

Make sure your Ollama server is active.

### Step 2: Start Streamlit

```bash
streamlit run app.py
```

Access the chatbot at [http://localhost:8501](http://localhost:8501)

---

## 💬 Usage

### Upload Documents

- Use the "Upload Documents" section in the UI to add files (PDF, CSV, DOCX, PPTX, TXT, MD).
- You'll get a success message after processing.

### Ask Questions

- Type your query in the input box.
- Click "Submit Query" and wait for the AI response.

### View Answers and Sources

- The chatbot will show:
  - **Answer** generated using local LLM.
  - **MCP Retrieval Result**: Raw chunk context used for generation.
  - **Simple Sources from Context**: Friendly filenames/chunk indicators.

---

## 🧩 Challenges Faced

- **Robust Document Parsing**: Required diverse format support and error handling.
- **Chunking Strategy**: Balancing chunk size with context quality.
- **FAISS Integration**: Managing embeddings and search accuracy.
- **Ollama Integration**: Prompt design and communication handling.
- **MCP Implementation**: Designing structured, traceable inter-agent messaging.
- **Streamlit Session Management**: Managing `st.session_state` for app continuity.

---

## 🚧 Future Enhancements

- Improved UI Feedback & Error Handling
- Hybrid Search (Vector + Keyword)
- Agent Self-Correction and Clarification
- Pluggable LLMs (OpenAI, Gemini, etc.)
- Multi-User Environment Support (auth + separate docs)

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
