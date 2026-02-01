# 💬 Portfolio Chatbot

An AI-powered chatbot that lets recruiters interact directly with my professional profile — asking about skills, projects, research, and experience in a natural conversation.

## 🚀 Live Demo

👉 [Chat with Nikhilesh](https://nikhileshportfoliochatbot.streamlit.app/)

---

## 🏗️ How It Works

```
Recruiter asks a question
        ↓
FAISS retrieves relevant resume chunks (MMR search)
        ↓
LangChain RunnableParallel formats context + passes question
        ↓
Groq (Llama 3.3 70B) generates a response in Nikhilesh's voice
        ↓
Streamlit streams the answer to the recruiter
```

### Key Components
- **RAG (Retrieval-Augmented Generation):** Grounds every answer in real resume data — no hallucinations.
- **MMR Retrieval:** Maximal Marginal Relevance ensures diverse and relevant chunks are retrieved.
- **Personality-Driven Prompting:** Responses sound like Nikhilesh — caring, confident, optimistic, and approachable.
- **Prioritized Experience Tiers:** When experience is asked, the bot surfaces the most relevant DS/AI/ML roles first.
- **Quick Action Buttons:** One-click access to Skills, Projects, Research, and Experience for a seamless recruiter experience.

---

## 📂 Project Structure

```
portfolio-chatbot/
├── app.py                  ← Streamlit chatbot (main app)
├── ingest.py               ← Embeds resume into FAISS vector DB
├── resume.txt              ← Master resume (plain text)
├── faiss_db/               ← FAISS vector store (pre-built)
├── requirements.txt        ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python** | Core language |
| **Streamlit** | UI & hosting |
| **LangChain** | RAG chain (RunnableParallel, PromptTemplate) |
| **FAISS** | Vector database for semantic search |
| **Sentence Transformers** | Resume embedding (all-MiniLM-L6-v2) |
| **Groq** | LLM inference (Llama 3.3 70B) |
| **HuggingFace** | Embeddings model |

---

## ⚙️ Setup Locally

**1. Clone the repo**
```bash
git clone https://github.com/nikhileshnarkhede/portfolio-chatbot.git
cd portfolio-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com)

**5. (Optional) Re-ingest resume**

If you update `resume.txt`, rebuild the vector DB:
```bash
python ingest.py
```

**6. Run the app**
```bash
streamlit run app.py
```

Opens at → `http://localhost:8501`

---

## 🌐 Deployment

Hosted on **Streamlit Community Cloud** — free tier.

- Repo: [github.com/nikhileshnarkhede/portfolio-chatbot](https://github.com/nikhileshnarkhede/portfolio-chatbot)
- Live: [portfolio-chatbot.streamlit.app](https://portfolio-chatbot.streamlit.app)

---

## 📌 Features

- ✅ Natural language conversation with recruiters
- ✅ RAG-powered — answers grounded in real resume only
- ✅ Personality-driven responses (caring, confident, soft tone)
- ✅ Quick action buttons for instant access
- ✅ All 10 projects listed with clickable portfolio links
- ✅ Prioritized experience tiers (DS/AI/ML roles shown first)
- ✅ Contact card with LinkedIn, GitHub, Portfolio links
- ✅ Dark-themed, clean UI

---

## 📬 Contact

| | |
|---|---|
| **Email** | narkhede.nikhilesh@gmail.com |
| **LinkedIn** | [linkedin.com/in/nikhileshnarkhede](https://www.linkedin.com/in/nikhileshnarkhede) |
| **GitHub** | [github.com/nikhileshnarkhede](https://github.com/nikhileshnarkhede) |
| **Portfolio** | [nikhileshnarkhede.github.io/portfolio](https://nikhileshnarkhede.github.io/portfolio/) |
