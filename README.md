# Banking Customer Support AI – Multi-Agent + RAG Demo

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DB_URL=sqlite:///support.db
LOG_LEVEL=INFO
```

## Knowledge Base (RAG)

Place your banking support docs as `.txt` or `.md` into:

```text
knowledge_base/
  faq_reset_password.txt
  faq_debit_card_blocking.md
  ...
```

Then build the embeddings index:

```bash
python -m app.rag.ingest
```

## Running the app

```bash
streamlit run ui/streamlit_app.py
```

- Queries with a **ticket number** use the ticket status flow.
- General **support questions** (no ticket number) are answered using RAG over the support documents.
