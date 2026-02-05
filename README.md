# 💬 Nikhilesh's Portfolio Chatbot

An intelligent, conversational AI chatbot powered by **Groq LLaMA 3.1 8B** and **Retrieval-Augmented Generation (RAG)** that lets recruiters and visitors interact with my professional profile naturally.

🔗 **Live Demo**: [https://nikhileshportfoliochatbot.streamlit.app/](https://nikhileshportfoliochatbot.streamlit.app/)

🌐 **Portfolio Website**: [https://nikhileshnarkhede.github.io/portfolio/](https://nikhileshnarkhede.github.io/portfolio/)

---

## ✨ Features

### 🤖 **Advanced Conversational AI**
- **First-person responses**: The chatbot speaks AS Nikhilesh, not about him
- **Human-like personality**: Caring, optimistic, confident, and easy to understand
- **Streaming responses**: Text appears word-by-word like ChatGPT with a cursor effect (█)
- **Natural follow-ups**: Understands context from previous messages
- **Blazing fast**: Uses Groq's fastest "instant" model for sub-second responses

### 🧠 **Enhanced RAG System**
- **Query expansion**: Automatically enhances queries with relevant keywords
  - Example: "skills" → "skills technical programming frameworks libraries"
- **MMR retrieval**: Retrieves 8 diverse document chunks for comprehensive answers
- **Semantic search**: FAISS vector database with sentence-transformers embeddings
- **Structured context**: Numbered sections help the LLM understand retrieved information

### 💾 **Smart Conversation Memory**
- **Short-term memory**: Remembers the last 3 exchanges (6 messages) in full detail
- **Progressive auto-summarization**: Compresses old messages after 12 messages
- **Infinite conversations**: Never hits token limits — chat indefinitely!
- **Context-aware**: Handles follow-ups like "tell me more", "what about that project?", "and the others?"

### 🎯 **User Experience**
- **Quick action buttons**: Pre-set questions (Skills, Projects, Research, Experience)
- **Contact card**: Direct links to LinkedIn, GitHub, portfolio
- **Professional UI**: Clean, dark-themed, recruiter-friendly interface
- **Error handling**: Graceful fallbacks for rate limits and API errors

### 🔍 **Debug Mode** (Expandable panels)
- **🔍 Enhanced Query**: See how queries are expanded
- **📚 Retrieved Context**: View document chunks used to answer
- **💬 Conversation Memory**: Check what's remembered + auto-summarization status

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Query Expansion│ ("skills" → "skills technical programming frameworks")
              └───────┬────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  FAISS Vector Search   │ (MMR: 8 chunks, fetch_k=20, λ=0.7)
         │  + Semantic Embeddings │
         └────────┬───────────────┘
                  │
                  ▼
        ┌──────────────────────┐
        │  Format & Structure  │ ([Section 1], [Section 2], ...)
        └──────────┬───────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Conversation Memory             │
    │  • Last 6 messages (full detail) │
    │  • Auto-summary (if >12 messages)│
    └──────────┬───────────────────────┘
               │
               ▼
      ┌─────────────────────┐
      │  Groq LLaMA 3.1 8B  │ (Streaming response with cursor)
      │  500K tokens/day    │
      └────────┬────────────┘
               │
               ▼
      ┌────────────────────┐
      │  Natural Response   │
      │  with Clickable URLs│
      └─────────────────────┘
```

---

## 🚀 Technology Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq LLaMA 3.1 8B Instant (500K tokens/day) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Framework** | LangChain + LangChain-Groq |
| **UI** | Streamlit |
| **Hosting** | Streamlit Cloud (Free tier) |
| **RAG Strategy** | MMR (Maximal Marginal Relevance) |
| **Memory** | Progressive auto-summarization |

---

## 📦 Installation & Setup

### **Prerequisites**
- Python 3.12+
- Groq API Key (free from [console.groq.com](https://console.groq.com))

### **1. Clone the Repository**
```bash
git clone https://github.com/nikhileshnarkhede/portfolio-chatbot.git
cd portfolio-chatbot
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Set Up Groq API Key**

Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Or set environment variable:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### **4. Build the Vector Database**
```bash
python ingest.py
```

This creates a FAISS index from `resume.txt` (only needed once, or when you update the resume).

### **5. Run Locally**
```bash
streamlit run app.py
```

The chatbot will open at `http://localhost:8501`

---

## 📝 Updating the Resume

To update the chatbot's knowledge:

1. Edit `resume.txt` with new information
2. Rebuild the vector database:
   ```bash
   python ingest.py
   ```
3. Restart the app

**Important**: Always include full URLs for projects in this format:
```
Project Name - Description
Built...
URL: https://example.com/project-page.html
```

---

## 🎨 Customization

### **Change the LLM Model**
Edit `app.py` (lines 12-16):
```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Change this
    api_key=GROQ_API_KEY,
    temperature=0.3,  # Adjust creativity (0.0-1.0)
)
```

**Available Groq models:**
- `llama-3.1-8b-instant` - 500K tokens/day, fastest
- `llama-3.3-70b-versatile` - 100K tokens/day, highest quality
- `llama-3.1-70b-versatile` - 100K tokens/day, balanced

### **Adjust Conversation Memory**
Change auto-summarization trigger in `app.py` (line 317):
```python
if message_count > 12:  # Trigger after 12 messages
```

### **Modify Retrieval Settings**
Tune RAG parameters in `app.py` (lines 24-27):
```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,           # Number of chunks (increase for more context)
        "fetch_k": 20,    # Candidate pool (higher = more diverse)
        "lambda_mult": 0.7  # Diversity (0=max diversity, 1=max relevance)
    }
)
```

### **Change Personality**
Edit the prompt template in `app.py` (lines 87-110):
```python
Your personality:
- Caring: You genuinely want the recruiter to feel heard...
- Optimistic: You see the bright side...
- Assertive: You own your achievements confidently...
```

---

## 🚀 Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `app.py`
5. Add your `GROQ_API_KEY` in **App settings → Secrets**:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
6. Deploy!

---

## 🔧 Project Structure

```
portfolio-chatbot/
├── app.py                    # Main Streamlit application (500 lines)
├── ingest.py                 # Vector DB builder
├── resume.txt                # Source resume with projects & URLs
├── faiss_db/                 # FAISS vector index (pre-built)
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python 3.12
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── .streamlit/
│   └── secrets.toml          # API keys (not committed)
└── venv/                     # Virtual environment (local only)
```

---

## 🧪 Testing Features

### **Test Short-Term Memory**
```
User: "What are your projects?"
Bot: [Lists 10 projects with URLs]
User: "Tell me more about the first one"
Bot: [Explains Job Application Assistant in detail]
User: "What technologies did you use for that?"
Bot: [References the same project - remembers context!]
```

### **Test Auto-Summarization**
1. Have a long conversation (13+ messages)
2. Click **💬 Conversation Memory (Debug)**
3. You'll see:
   ```
   [Earlier conversation summary]
   The recruiter asked about Nikhilesh's ML projects and Python skills.
   
   [Recent conversation]
   Recruiter: Tell me about your research
   Nikhilesh: [recent response]
   ```
4. Notice the 🔄 icon confirming auto-summarization is active

### **Test Query Expansion**
1. Ask: "What are your skills?"
2. Click **🔍 Enhanced Query (Debug)**
3. See: 
   ```
   Original: What are your skills?
   Enhanced: What are your skills technical skills programming frameworks libraries
   ```

### **Test Streaming**
- Type any question
- Watch text appear word-by-word with cursor (█)
- Feels like ChatGPT!

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Average Response Time** | 1-3 seconds (streaming starts <1s) |
| **Token Limit** | 500K/day (enough for 200+ conversations) |
| **Memory Efficiency** | Auto-summarizes after 12 messages |
| **Concurrent Users** | 100+ (Groq free tier) |
| **Uptime** | 99.9% (Streamlit Cloud) |
| **Cost** | $0 (free Groq API + Streamlit Cloud) |

---

## 🎯 Use Cases

✅ **For Recruiters**: Quick way to learn about skills, projects, and experience  
✅ **For Hiring Managers**: Ask specific technical questions about ML/AI work  
✅ **For Students**: Example of production RAG system with memory  
✅ **For Developers**: Reference implementation for portfolio chatbots  
✅ **For Researchers**: See progressive summarization in action  

---

## 🛠️ Troubleshooting

### **Rate Limit Errors**
The chatbot uses **llama-3.1-8b-instant** with 500K tokens/day limit. If you hit the limit:
- **Wait 24 hours** for reset
- **Upgrade to Groq paid tier** ($0.20/M tokens)
- **Contact me directly**: narkhede.nikhilesh@gmail.com

### **Empty or Generic Responses**
- Check if `resume.txt` exists and has content
- Rebuild vector DB: `python ingest.py`
- Verify the topic exists in `resume.txt`
- Try asking more specific questions

### **Slow Responses**
- First response may take 3-5s (cold start)
- Subsequent responses should be 1-2s
- Check Groq API status if consistently slow
- Model is "instant" — should be very fast

### **"Context not found" in Debug Panel**
- Query might be too vague or off-topic
- Try more specific questions
- Check if the topic exists in `resume.txt`
- Example: "machine learning" works, "cooking" won't

### **Streamlit Cache Issues**
If model changes don't apply:
```bash
# Clear cache and restart
streamlit cache clear
streamlit run app.py
```

---

## 🔄 Recent Updates

**v2.0 (Current)**
- ✅ Switched to LLaMA 3.1 8B Instant (500K tokens/day)
- ✅ Added streaming responses with cursor effect
- ✅ Implemented conversation memory with auto-summarization
- ✅ Enhanced RAG with query expansion
- ✅ Added graceful error handling
- ✅ Debug mode with expandable panels

**v1.0**
- Initial release with LLaMA 3.3 70B
- Basic RAG implementation
- Static responses

---

## 📜 License

MIT License - Feel free to fork and customize for your own portfolio!

---

## 🤝 Contributing

Found a bug or have a feature idea? 
- Open an issue on GitHub
- Submit a pull request
- Email me: narkhede.nikhilesh@gmail.com

---

## 🙏 Acknowledgments

Built with amazing open-source tools:
- **Groq** - Lightning-fast LLM inference
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search (Meta AI)
- **Streamlit** - Beautiful web apps
- **Sentence Transformers** - Semantic embeddings

---

## 📧 Contact

**Nikhilesh Narkhede**  
📧 narkhede.nikhilesh@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/nikhileshnarkhede)  
💻 [GitHub](https://github.com/nikhileshnarkhede)  
🌐 [Portfolio](https://nikhileshnarkhede.github.io/portfolio/)

---

**Built with ❤️ using Groq, LangChain, and Streamlit**

*Last updated: February 2026*
