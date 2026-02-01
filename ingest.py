from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load resume
loader = TextLoader("resume.txt", encoding="utf-8")
docs = loader.load()

# Larger chunks to keep projects/experience together
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedder)
db.save_local("./faiss_db")

print(f"✅ Stored {len(chunks)} chunks into faiss_db/")

# --- Debug: print chunks so you can see what's being stored ---
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk.page_content[:150])
