import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Config ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Force model recreation (no caching)
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # 500K tokens/day - 5x more than 70b!
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

print(f"🤖 Using model: {llm.model_name}")  # Debug log

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("./faiss_db", embedder, allow_dangerous_deserialization=True)

# --- MMR Retriever (enhanced) ---
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
)

# --- Query Enhancement ---
def enhance_query(query: str) -> str:
    """Expand query with relevant keywords for better retrieval"""
    query_lower = query.lower()
    
    # Add context keywords based on query intent
    if any(word in query_lower for word in ["skill", "technolog", "tool", "language"]):
        return f"{query} technical skills programming frameworks libraries"
    elif any(word in query_lower for word in ["project", "built", "developed", "created"]):
        return f"{query} projects applications systems built developed"
    elif any(word in query_lower for word in ["experience", "work", "job", "role", "position"]):
        return f"{query} experience work employment roles responsibilities"
    elif any(word in query_lower for word in ["research", "publication", "paper", "study"]):
        return f"{query} research publications papers studies machine learning"
    elif any(word in query_lower for word in ["education", "degree", "university", "study"]):
        return f"{query} education degree university certification"
    else:
        return query

# --- Format docs with better structure ---
def format_docs(retrieved_docs):
    """Format retrieved documents with clear sections"""
    if not retrieved_docs:
        return "No relevant information found."
    
    formatted = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.strip()
        formatted.append(f"[Section {i}]\n{content}")
    
    return "\n\n".join(formatted)

# --- Format conversation history with summarization ---
def format_chat_history(messages):
    """Format chat history - uses summary + recent messages"""
    if not messages or len(messages) == 0:
        # Check if we have a summary from earlier
        if st.session_state.conversation_summary:
            return f"[Earlier conversation summary]\n{st.session_state.conversation_summary}\n\n[Current conversation]\nNo messages yet."
        return "No previous conversation."
    
    # Build the history
    history_parts = []
    
    # Add summary if it exists
    if st.session_state.conversation_summary:
        history_parts.append(f"[Earlier conversation summary]\n{st.session_state.conversation_summary}\n")
    
    # Add recent messages in full detail
    if messages:
        history_parts.append("[Recent conversation]")
        for msg in messages:
            role = "Recruiter" if msg["role"] == "user" else "Nikhilesh"
            history_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_parts)

# --- Prompt Template ---
prompt = PromptTemplate(
    template="""
        You are Nikhilesh, speaking directly to a recruiter about your own background, skills, and experience.
        You are NOT a third-party assistant — you ARE Nikhilesh. Use "I", "my", "me" naturally.

        Your personality:
        - Caring: You genuinely want the recruiter to feel heard and valued. Acknowledge their questions warmly before diving in.
        - Optimistic: You see the bright side — frame your experience as growth, learning, and impact.
        - Assertive: You own your achievements confidently. Don't undersell yourself. State what you've done clearly.
        - Confident: You speak with certainty. No hedging words like "maybe", "I think", or "possibly" unless truly unsure.
        - Soft tone: Your confidence doesn't come across as aggressive. It feels approachable, genuine, and easy to connect with.
        - Easy to understand: Keep it simple. Avoid unnecessary jargon. If you use a technical term, briefly explain what it means.

        **IMPORTANT: Use the conversation history below to maintain context and answer follow-up questions naturally.**

        Previous Conversation:
        {chat_history}

        Tone examples:
        - Instead of: "I have experience in machine learning."
        - Say: "Machine learning has been a big part of my journey — I've built models that actually solved real problems, like predicting material properties with over 99% accuracy."

        - Instead of: "I worked on a RAG project."
        - Say: "One project I really enjoyed was building a conversational AI system — it lets users ask questions about videos and get accurate answers instantly. Super rewarding to build end to end."

        Rules:
        - Answer ONLY from the provided resume context. Never make up information.
        - Use the conversation history to understand follow-up questions like "tell me more", "what about that project?", "and the others?"
        - If the context doesn't cover the question, say it warmly:
          "That's a great question! I don't have that detail here right now, but I'd love to chat about it directly — feel free to reach out!"
        - Keep answers focused but not too short. Give enough detail to impress, but don't overwhelm.
        - End with a warm, inviting line when appropriate — like offering to elaborate or connect.
        - IMPORTANT: Whenever you talk about a project, ALWAYS include the Repo Link or URL from the context.
          Format it as a clickable markdown link like this: [Check it out here](URL)
          Place it naturally at the end of the project description.
        - If multiple projects are mentioned, include each project's URL separately.
        - Only share URLs that actually exist in the context. Never make up or guess URLs.

        About Me (use this when asked who you are or about your background):
        I am a Master's student in Data Science. I hold a Bachelor's degree in Mechanical Engineering and an Advanced Certification in Artificial Intelligence and Machine Learning from IIT Kanpur. I bring a strong academic foundation and a deep enthusiasm for statistical modeling, mathematics, and using data to solve real-world problems. My approach spans the entire data science lifecycle — from data ingestion and preprocessing to model deployment and performance optimization — and is grounded in both theory and practical application. I excel at translating complex, ambiguous domain problems into clear, impactful machine learning tasks, and I specialize in building interpretable, scalable, and production-ready models that deliver measurable results. I am equally adept at communicating data-driven insights to both technical and non-technical audiences. I have a strong interest in research and enjoy working at the intersection of data, experimentation, and domain expertise to develop innovative, AI-driven solutions. For me, data science is more than just a discipline — it is a powerful tool for transforming ideas into actionable strategies and real-world impact.

        All My Projects (ALWAYS use this list when projects are asked):
        1. Job Application Assistant — LLM-Powered AI Application
           Built an end-to-end AI app that automates resume customization, cover letter generation using LLMs and prompt engineering.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/job-application-assistant.html)

        2. Chat with YouTube Videos — Retrieval-Augmented Conversational AI
           Built a full RAG system that lets users ask questions about YouTube videos and get accurate, context-aware answers using FAISS, LangChain, and LLMs.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/chat-youtube-videos.html)

        3. Real-Time Market Momentum & Sentiment Prediction — NLP + LSTM System
           Built a deep learning framework combining financial news sentiment (NLP) with LSTM networks to predict S&P 500 market momentum in real time.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/market-sentiment-prediction.html)

        4. NLP Systems for Text Classification & Sentiment Analysis
           Built NLP pipelines for spam detection, sentiment classification, and complaint resolution using word embeddings (GloVe, FastText) and LSTM models.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/nlp-text-classification.html)

        5. Applied Machine Learning for Credit Risk & Loan Default Prediction
           Built supervised ML models to predict payday loan default risk. LightGBM and CatBoost were top performers, also benchmarked against a feedforward neural network.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/credit-risk-prediction.html)

        6. High-Performance Deep Learning for Music Genre Classification
           Built a deep learning pipeline classifying music genres from raw audio using spectrograms, CNN, RNN, and LSTM with parallel computing for speed.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/music-genre-classification.html)

        7. Applied Machine Learning for Business & Industrial Forecasting
           Built multiple ML pipelines across business, finance, industrial, and healthcare domains — including customer subscription prediction, vehicle emissions forecasting, and interest rate prediction.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/ml-business-forecasting.html)

        8. End-to-End Computer Vision Systems for Image Classification & Recognition
           Built CNN-based pipelines for face detection (MTCNN + VGGFace), handwritten digit classification (MNIST), and medical image diagnosis using transfer learning.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/computer-vision.html)

        9. Supply Chain Tracker — Data-Driven Analytics & Web Application
           Built a full-stack supply chain app with interactive dashboards (Streamlit + Plotly), role-based access control, and CSV data pipelines.
           [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/supply-chain-tracker.html)

        10. Data Engineering & Visualization Applications
            Built interactive data-driven apps including real-time dashboards, multi-view visualizations, and force simulations using Streamlit, Plotly, and D3.js.
            [View Project](https://nikhileshnarkhede.github.io/portfolio/projects/data-engineering-visualization.html)

        Also my published research:
        - Prediction of Mechanical and Fracture Properties of Lightweight Polyurethane Composites Using Machine Learning Methods
          [Read Paper](https://doi.org/10.3390/jcs9060271)

        All My Experience (ALWAYS use this prioritized list when experience is asked.
        Show Tier 1 first with detail, then Tier 2 briefly, then mention Tier 3 as supporting background):

        === TIER 1: Core Data Science / AI / ML Roles ===

        1. University of Massachusetts Dartmouth — Research Assistant (Part-Time) | Jun 2025 – Dec 2025
           This is my current and most impactful role. I developed a physics-aware ML and optimization framework to predict mechanical and fracture properties of 3D printed materials. I used Symbolic Regression to extract interpretable equations, built deep neural network surrogate models, and implemented NSGA-II multi-objective optimization to find the best manufacturing parameters. I also generated synthetic datasets to work around limited experimental data. Achieved strong predictive results and created scientific visualizations like radial and parallel coordinate plots.
           Skills: Python, ML, Deep Learning, NSGA-II, Feature Engineering, Scientific Visualization

        2. Deep Learning Researcher (Math & Computational Consultant) — Contract | Feb 2025 – May 2025
           Built an end-to-end ML pipeline to predict properties of polyurethane composites. I trained and optimized ANN and DNN models with extensive hyperparameter tuning and achieved R² > 0.99 — outperforming traditional approaches. I also extrapolated material behavior beyond observed data by over 40%, showing strong generalization. This work led to a peer-reviewed publication in Journal of Composite Science (2025).
           Skills: Python, Deep Learning, Keras, TensorFlow, Data Augmentation, Hyperparameter Optimization

        3. Tech-Neo Publications LLP — Machine Learning Intern | Jul 2023 – May 2024
           Worked on ML models for materials and engineering research. Processed datasets for regression tasks, validated ML results against benchmarks, and supported research publication workflows.
           Skills: Python, ML Fundamentals, Pandas, NumPy, Research Methodology

        === TIER 2: Roles with Strong ML / Data / AI Components ===

        4. Bajaj Auto Ltd — Project Trainee (Predictive Maintenance) | Mar 2022 – Jun 2022
           Applied data analysis and ML fundamentals to predict equipment failures in automotive manufacturing. Worked with condition monitoring data to build maintenance forecasting strategies.
           Skills: Predictive Maintenance, Data Analysis, Python, Statistics, Industrial Analytics

        5. Abhiyantrix Academy — Technical Member (AI) | May 2020 – May 2021
           Explored ML concepts like supervised learning, regression, and classification. Applied linear algebra, probability, and optimization to AI model formulation.
           Skills: ML Fundamentals, Mathematics for AI, Python, Linear Algebra

        6. Code Karo Yaaro — Coding Tutor (ML / DL / Math for AI) | Sep 2021 – Nov 2021
           Taught Python, ML, and DL fundamentals. Created learning content on math foundations for AI like gradients, vectors, and optimization.
           Skills: Python, ML/DL Teaching, Mathematics for AI, Communication

        === TIER 3: Supporting Engineering Background ===

        7. DRDO (Armament Research & Development Establishment) — Research Assistant Intern | Sep 2022 – Mar 2023
           Worked on propulsion system design, CFD simulations, and automated calculations using Python and MATLAB. Strong engineering and simulation background.
           Skills: Python, MATLAB, CFD, Ansys, Mathematical Modeling

        8. Artenal — CAD Engineer Intern (Computer Vision Collaboration) | Jan 2022 – Jun 2022
           Collaborated with CV engineers on robotics projects. Designed end effectors for vision-based robotic manipulation and contributed to an autonomous apple-harvesting robot concept.
           Skills: Computer Vision (Applied), Robotics, CAD, SOLIDWORKS


        Resume Context:
        {context}

        Question: {question}
    """,
    input_variables=["context", "question", "chat_history"]
)

# --- RAG Chain with Query Enhancement and Chat History ---
def get_context(inputs):
    """Get context by enhancing query and retrieving docs"""
    enhanced = enhance_query(inputs["question"])
    docs = retriever.invoke(enhanced)
    return format_docs(docs)

parallel_chain = RunnableParallel({
    "context": RunnableLambda(get_context),
    "question": lambda x: x["question"],
    "chat_history": lambda x: x["chat_history"]
})

parser = StrOutputParser()
rag_chain = parallel_chain | prompt | llm | parser

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with Nikhilesh", page_icon="💬")

st.markdown("""
<style>
    .main {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .stChatInput input {
        background-color: #1a1d2e;
        color: #fff;
        border: 1px solid #333;
    }
    h1 {
        color: #7c83fd;
        text-align: center;
    }
    .caption-text {
        text-align: center;
        color: #888;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with Nikhilesh")

# --- Contact Card (Top Middle) ---
st.markdown("""
<div style="text-align: center; padding: 15px 10px; margin-bottom: 10px;">
    <h2 style="color: #e0e0e0; margin-bottom: 4px; font-size: 22px;">Nikhilesh Narkhede</h2>
    <p style="color: #aaa; font-size: 13px; margin: 0;">US Work Authorised &nbsp;|&nbsp; +1 508-509-3697</p>
    <p style="color: #aaa; font-size: 13px; margin: 4px 0 10px 0;">narkhede.nikhilesh@gmail.com</p>
    <div style="display: flex; justify-content: center; gap: 18px;">
        <a href="https://www.linkedin.com/in/nikhileshnarkhede" target="_blank" style="color: #7c83fd; text-decoration: none; font-size: 14px;">🔗 LinkedIn</a>
        <a href="https://github.com/nikhileshnarkhede" target="_blank" style="color: #7c83fd; text-decoration: none; font-size: 14px;">🐙 GitHub</a>
        <a href="https://nikhileshnarkhede.github.io/portfolio/" target="_blank" style="color: #7c83fd; text-decoration: none; font-size: 14px;">🌐 Portfolio</a>
    </div>
</div>
<hr style="border-color: #2a2d3e; margin: 10px 0;">
""", unsafe_allow_html=True)
st.markdown('<p class="caption-text">Ask me about my skills, projects, research, and experience.</p>', unsafe_allow_html=True)

# Show rate limit info
with st.expander("💡 About this chatbot", expanded=False):
    st.info("""
    **Powered by Groq LLaMA 3.1 8B Instant** (500K tokens/day)
    
    Features:
    - ✅ 5x higher limit than before
    - ✅ Blazing fast responses
    - ✅ Streaming responses
    - ✅ Conversation memory
    - ✅ Auto-summarization (prevents token overflow)
    - ✅ Query expansion for better answers
    
    Questions? Email: narkhede.nikhilesh@gmail.com
    """)

# --- Quick Action Buttons ---
button_questions = {
    "🛠️ Skills": "What are all your technical skills?",
    "📁 Projects": "Tell me about all the projects you have built.",
    "🔬 Research": "Tell me about your research work.",
    "💼 Experience": "Walk me through your work experience."
}

# --- Session State (must be initialized BEFORE buttons check it) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "triggered_question" not in st.session_state:
    st.session_state.triggered_question = None
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# --- Auto-summarize old messages when conversation gets long ---
def auto_summarize_if_needed():
    """Automatically summarize old conversation to save tokens"""
    message_count = len(st.session_state.messages)
    
    # Trigger summarization when we have more than 12 messages (6 exchanges)
    if message_count > 12:
        # Get messages to summarize (all except last 6 - keep 3 recent exchanges)
        messages_to_summarize = st.session_state.messages[:-6]
        
        # Build conversation text
        conv_text = ""
        for msg in messages_to_summarize:
            role = "Recruiter" if msg["role"] == "user" else "Nikhilesh"
            conv_text += f"{role}: {msg['content']}\n\n"
        
        # Create summary using Groq
        summary_prompt = f"""Briefly summarize this conversation between a recruiter and Nikhilesh in 2-3 sentences.
Focus on: topics discussed, projects mentioned, skills asked about.

Conversation:
{conv_text}

Brief summary:"""
        
        try:
            summary = llm.invoke(summary_prompt)
            st.session_state.conversation_summary = summary.content if hasattr(summary, 'content') else str(summary)
            # Remove the summarized messages, keep only recent ones
            st.session_state.messages = st.session_state.messages[-6:]
        except Exception as e:
            # If summarization fails, just truncate old messages
            st.session_state.conversation_summary = "Earlier conversation covered various topics about Nikhilesh's background."
            st.session_state.messages = st.session_state.messages[-6:]

# Only show buttons if chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    <style>
        .button-row {
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin: 16px 0 20px 0;
        }
        .quick-btn {
            background: linear-gradient(135deg, #1e2140, #2a2d4a);
            border: 1px solid #3a3f6b;
            color: #c8cce8;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .quick-btn:hover {
            background: linear-gradient(135deg, #2a2d4a, #3a3f6b);
            border-color: #7c83fd;
            color: #fff;
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(124, 131, 253, 0.3);
        }
    </style>
    <div class="button-row">
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (label, question) in enumerate(button_questions.items()):
        with cols[i]:
            if st.button(label, key=f"btn_{i}", use_container_width=True,
                         help=question):
                st.session_state.triggered_question = question

    st.markdown("</div>", unsafe_allow_html=True)

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Ask something about my background...")

# --- Pick up triggered question from buttons ---
if st.session_state.triggered_question and not user_input:
    user_input = st.session_state.triggered_question
    st.session_state.triggered_question = None

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Invoke RAG chain with streaming ---
    with st.chat_message("assistant"):
        # Build input with chat history
        chain_input = {
            "question": user_input,
            "chat_history": format_chat_history(st.session_state.messages)
        }
        
        # Show enhanced query in debug mode
        enhanced = enhance_query(user_input)
        if enhanced != user_input:
            with st.expander("🔍 Enhanced Query (Debug)", expanded=False):
                st.caption(f"Original: {user_input}")
                st.caption(f"Enhanced: {enhanced}")
        
        # Retrieve context
        docs = retriever.invoke(enhanced)
        context = format_docs(docs)
        
        # Show retrieved context in debug mode
        with st.expander("📚 Retrieved Context (Debug)", expanded=False):
            st.text(context[:1000] + "..." if len(context) > 1000 else context)
        
        # Show chat history in debug mode
        if len(st.session_state.messages) > 0 or st.session_state.conversation_summary:
            with st.expander("💬 Conversation Memory (Debug)", expanded=False):
                st.text(chain_input["chat_history"])
                if st.session_state.conversation_summary:
                    st.info(f"🔄 Auto-summarization active! Keeping last 6 messages + summary to save tokens.")
        
        # Generate answer with streaming and error handling
        answer_placeholder = st.empty()
        full_answer = ""
        
        try:
            for chunk in rag_chain.stream(chain_input):
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "█")  # Cursor effect
            
            # Final answer without cursor
            answer_placeholder.markdown(full_answer)
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limit errors
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("⏱️ **Rate Limit Reached**")
                st.warning("""
                The chatbot has reached Groq's free tier limit (100k tokens/day).
                
                **Options:**
                1. ⏰ **Wait 30 minutes** and try again
                2. 💳 **Upgrade to Groq Dev Tier** at [console.groq.com/settings/billing](https://console.groq.com/settings/billing)
                3. 📧 **Email me directly** at narkhede.nikhilesh@gmail.com
                
                This usually happens during high traffic. Sorry for the inconvenience!
                """)
                full_answer = "I've hit my daily token limit. Please try again in 30 minutes or contact me directly!"
                
            # Handle other API errors
            elif "api" in error_msg.lower() or "connection" in error_msg.lower():
                st.error("🔌 **API Connection Error**")
                st.warning("""
                There was a problem connecting to the AI service.
                
                **Options:**
                1. 🔄 **Refresh the page** and try again
                2. 📧 **Contact me directly** at narkhede.nikhilesh@gmail.com
                """)
                full_answer = "I'm having trouble connecting right now. Please try refreshing the page or contact me directly!"
                
            # Generic error
            else:
                st.error("❌ **Unexpected Error**")
                st.warning(f"""
                Something went wrong. Here's the error:
                
                ```
                {error_msg[:200]}
                ```
                
                Please contact me at narkhede.nikhilesh@gmail.com
                """)
                full_answer = "I encountered an error. Please contact me directly to discuss your questions!"

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    
    # Auto-summarize if conversation is getting long
    auto_summarize_if_needed()
