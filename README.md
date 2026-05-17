---
title: HR Policy Assistant
emoji: 🤝
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered HR Policy chatbot using RAG + Llama 3.1
---

# 🤝 HR Policy Assistant

> Ask any question about your company's HR policies and get instant, accurate answers — grounded in your actual policy documents.

---

## 💡 What Does This Do?

Most employees don't read HR policy manuals. They're long, boring, and hard to navigate. This assistant changes that.

You just **type your question in plain English** — and the AI finds the right answer from your HR documents, tells you exactly which document and page it came from, and responds in seconds.

**Examples of questions you can ask:**
- *"How many casual leaves do I get per year?"*
- *"What is the work from home policy?"*
- *"How do I apply for maternity leave?"*
- *"What expenses are covered under travel reimbursement?"*
- *"What is the notice period if I resign?"*

---

## 🧠 How It Works (Simple Version)

```
You ask a question
       ↓
AI searches through your HR policy documents
       ↓
Finds the most relevant sections
       ↓
Generates a clear answer based on those sections
       ↓
Shows you exactly which document and page it used
```

This approach is called **RAG (Retrieval-Augmented Generation)** — it means the AI never makes things up. Every answer comes directly from your documents.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Source Citations** | Every answer shows which document and page it came from |
| 💬 **Multi-turn Chat** | Ask follow-up questions naturally |
| ⚡ **Fast Responses** | Powered by Groq — answers in under 2 seconds |
| 🔒 **Accurate** | AI only answers from your documents, never guesses |
| 🎨 **Beautiful UI** | Clean dark interface, easy to use |

---

## 🛠️ Built With

| Technology | Purpose |
|---|---|
| **Llama 3.1 8B** | The AI brain that reads and answers |
| **Groq API** | Makes Llama 3.1 extremely fast (free tier available) |
| **ChromaDB** | Stores and searches document chunks |
| **sentence-transformers** | Converts text to searchable vectors |
| **FastAPI** | REST API backend |
| **Streamlit** | The web interface you're using |

---

## 🚀 How to Run This Yourself

### Step 1 — Get a free Groq API key
Go to [console.groq.com](https://console.groq.com/keys) and sign up for free.

### Step 2 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-rag-chatbot.git
cd hr-rag-chatbot
```

### Step 3 — Add your API key
```bash
cp .env.example .env
# Open .env and paste your GROQ_API_KEY
```

### Step 4 — Add your HR documents
```bash
# Drop your PDF or DOCX files here:
data/hr_policies/
```

### Step 5 — Install and run
```bash
pip install -r requirements.txt
python ingest.py        # loads your documents (run once)
streamlit run app.py    # starts the app
```

Open [http://localhost:8501](http://localhost:8501) and start chatting!

---

## 🐳 Run with Docker (even easier)
```bash
docker-compose up --build
```
That's it — everything starts automatically including document ingestion.

---

## 🔐 Using on HuggingFace Spaces

To use your own HR documents and API key on this Space:

1. **Fork this Space** (click Fork at the top)
2. Go to **Settings → Variables and Secrets**
3. Add a secret: `GROQ_API_KEY` = your key
4. Add your PDF files to `data/hr_policies/` and push
5. The app will automatically ingest them on first startup

---

## 📁 What Files Does It Need?

Put your HR policy documents in the `data/hr_policies/` folder.

**Supported formats:**
- PDF (`.pdf`) — page numbers are preserved in citations
- Word documents (`.docx`)
- Plain text (`.txt`)
- Markdown (`.md`)

---

## ⚙️ Configuration Options

You can customize behavior by changing these in your `.env` file:

| Setting | Default | What It Does |
|---|---|---|
| `LLM_MODEL` | `llama-3.1-8b-instant` | Which AI model to use |
| `CHUNK_SIZE` | `500` | How large each document chunk is |
| `TOP_K_RESULTS` | `5` | How many chunks to retrieve per question |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum relevance score (0-1) |

**Want better answers?** Switch to a more powerful model:
```
LLM_MODEL=llama-3.3-70b-versatile
```

---

## ❓ Frequently Asked Questions

**Q: Will it make up answers?**
No. If the answer isn't in your HR documents, it will say so and suggest contacting HR directly.

**Q: Is my data safe?**
Your documents stay in your Space. Only the text of retrieved chunks is sent to Groq's API to generate the answer.

**Q: What if I update my HR documents?**
Re-run `python ingest.py --reset` to rebuild the knowledge base with the updated documents.

**Q: Can it handle multiple documents?**
Yes — drop as many PDFs as you want into `data/hr_policies/`. It searches across all of them.

---

## 📝 License

MIT License — free to use, modify, and deploy.

---

*Built with ❤️ using open source AI tools*
