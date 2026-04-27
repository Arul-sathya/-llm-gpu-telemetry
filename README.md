# FinAgent 🤖📈
### Agentic RAG System for SEC Filing Intelligence

A production-grade financial analyst AI that reasons over SEC 10-K/10-Q filings to answer questions, extract quantitative signals, compare companies, and backtest NLP sentiment signals against stock returns.

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that grounds an LLM's answers in real documents rather than relying on its training data alone.

```
Without RAG:  Question → LLM (guesses from memory) → Answer
With RAG:     Question → Fetch relevant documents → LLM reads them → Grounded Answer
```

Instead of hallucinating financial figures, the model retrieves actual text from Apple's 10-K and answers based on what's written there — with citations.

---

## What makes FinAgent *agentic*?

Basic RAG does one thing: embed → retrieve → answer. FinAgent goes further — it uses an **LLM agent loop** that plans and executes multiple steps:

```
User question
    ↓
Agent plans: "I should search filings, then extract signals"
    ↓
Tool: search_filings(query="revenue growth", tickers=["AAPL"])
    ↓
Tool: extract_signals(ticker="AAPL")
    ↓
Tool: compare_companies(topic="AI investment", tickers=["AAPL","NVDA"])
    ↓
Tool: final_answer("Based on filings from...")
```

1. **Plans** its approach before retrieving anything
2. **Chooses tools** dynamically based on the question
3. **Iterates** — uses results from one step to inform the next
4. **Synthesizes** a final answer with filing citations

---

## Features

| Feature | Description |
|---|---|
| **Agentic planning** | LLM decides which tools to call and in what order |
| **Multi-company RAG** | Compare Apple, Nvidia, Microsoft, Tesla, Meta in one query |
| **Signal extraction** | Structured JSON: revenue, margins, EPS from raw filing text |
| **Contradiction detection** | Find where management changed their narrative over time |
| **Sentiment backtest** | Correlate filing sentiment scores with 30-day post-earnings returns |
| **Streamlit UI** | Full web interface with charts, tables, and query history |

---

## Stack

| Layer | Technology | Cost |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Free (100k tokens/day) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free (runs locally) |
| Vector DB | FAISS | Free (runs locally) |
| Filing data | SEC EDGAR API | Free (public) |
| Market data | yfinance | Free |
| UI | Streamlit | Free |

**Total cost to run: $0**

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finagent.git
cd finagent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open http://localhost:8501, enter your free [Groq API key](https://console.groq.com), load tickers, and ask questions.

---

## Deploy to Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/spaces
2. Select **Streamlit** as the SDK
3. Upload `app.py`, `rag_engine.py`, `requirements.txt`
4. Add `GROQ_API_KEY` as a Space secret
5. Done — get a public shareable URL

---

## Supported companies

| Ticker | Company |
|---|---|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corp. |
| NVDA | Nvidia Corp. |
| TSLA | Tesla Inc. |
| META | Meta Platforms |
| GOOGL | Alphabet Inc. |
| AMZN | Amazon.com Inc. |

---

## Agent tools

| Tool | What it does |
|---|---|
| `search_filings` | Dense FAISS retrieval over indexed SEC filings |
| `extract_signals` | Structured JSON metric extraction (revenue, margins, EPS) |
| `compare_companies` | Cross-company topic synthesis |
| `get_price_data` | Historical returns via yfinance |
| `backtest_sentiment` | Sentiment score → 30d return correlation |
| `final_answer` | Synthesized response with citations |

---

## Resume bullet

> Built **FinAgent** — an agentic RAG system over SEC 10-K/10-Q filings (FAISS + sentence-transformers + Groq); LLM agent dynamically plans and executes retrieval, signal extraction, and multi-company comparison; includes backtested NLP sentiment signals correlated against 30-day post-earnings returns. Deployed as Streamlit app on Hugging Face Spaces.
