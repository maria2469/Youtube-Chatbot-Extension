import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

# -----------------------------
# Bootstrap
# -----------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY is missing. Add it to your environment or .env file.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

# -----------------------------
# FastAPI app + CORS (allow all for now; tighten later)
# -----------------------------
app = FastAPI(title="YouTube RAG Chatbot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "chrome-extension://gfcfobnonmfkpfpgaebpajpgpbjhbpgc",  # your extension ID
        "http://localhost:5173", ],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    video_id: str
    query: str
    prefer_langs: Optional[List[str]] = None  # e.g., ["en", "en-US", "en-GB"]

class QueryResponse(BaseModel):
    answer: str

# -----------------------------
# Helpers
# -----------------------------

def fetch_transcript_text(video_id: str, prefer_langs: Optional[List[str]] = None) -> str:
    """Fetch transcript text. Tries preferred languages first, then falls back to common ones.
    Returns full transcript as a single string.
    """
    ytt_api = YouTubeTranscriptApi()

    # 1) Try preferred languages (default to English variants)
    langs = prefer_langs or ["en", "en-US", "en-GB"]
    try:
        fetched = ytt_api.fetch(video_id, languages=langs)
        return " ".join(snippet.text for snippet in fetched)
    except Exception:
        pass

    # 2) Try a broader set of common languages (add more if your audience needs)
    fallback_langs = ["hi", "ur", "es", "fr", "de", "ar", "pt", "id"]
    try:
        fetched = ytt_api.fetch(video_id, languages=langs + fallback_langs)
        return " ".join(snippet.text for snippet in fetched)
    except TranscriptsDisabled as e:
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video.") from e
    except Exception as e:
        # Final attempt: surface a clear error
        raise HTTPException(status_code=404, detail=f"No transcript found for video_id={video_id} in supported languages.") from e


def build_chain(chunks: List[str]):
    """Build a retrieval-augmented generation chain for a given set of text chunks."""
    # Vector store per request (stateless API). If you need persistence, swap to a hosted DB/vector store.
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template=(
            "You are an expert assistant. Answer the user's question using ONLY the provided transcript snippets.\n"
            "If the answer isn't present, say you can't find it in the video. Be concise.\n\n"
            "Question:\n{query}\n\n"
            "Transcript snippets:\n{text}\n\n"
            "Answer:"
        ),
        input_variables=["query", "text"],
    )

    model = ChatGroq(model="llama3-8b-8192", temperature=0)
    parser = StrOutputParser()

    parallel = RunnableParallel({
        "text": retriever | RunnableLambda(format_docs),
        "query": RunnablePassthrough(),
    })

    chain = parallel | prompt | model | parser
    return chain


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_video(body: QueryRequest, request: Request):
    # Debug logging
    print("🔹 Request method:", request.method)
    print("🔹 Request headers:", dict(request.headers))
    print("🔹 Raw body received from frontend:", await request.body())
    print("🔹 Parsed JSON (Pydantic model):", body.dict())

    # 1) Get transcript
    try:
        transcript = fetch_transcript_text(body.video_id, body.prefer_langs)
        print("✅ Transcript fetched successfully.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript fetch failed: {str(e)}") from e

    if not transcript.strip():
        raise HTTPException(status_code=404, detail="Empty transcript.")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)
    print(f"✅ Transcript split into {len(chunks)} chunks.")

    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to split transcript.")

    # 3) Build RAG chain and invoke
    try:
        chain = build_chain(chunks)
        answer = chain.invoke(body.query)
        print("✅ Query processed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM/RAG failure: {str(e)}") from e

    return QueryResponse(answer=answer)


# -----------------------------
# Local dev entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
