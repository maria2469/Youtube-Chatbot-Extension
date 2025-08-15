# 🎬 YouTube Chatbot Extension

AI-powered chatbot for YouTube videos — ask questions, get instant answers, and learn faster.

---

## 🚀 Problem It Solves
Watching long YouTube videos just to find one piece of information is frustrating.  
This extension lets you **ask a question directly about the video you're watching** and get instant, AI-generated answers from its transcript — saving you **time and effort**.

---

## 🚀 Features
✅ **AI-Powered Q&A** — Ask any question about the current YouTube video.  
✅ **Real-Time Transcript Analysis** — Parses video captions for context-aware responses.  
✅ **Modern UI** — Clean and responsive popup interface.  
✅ **Voice Support** — Ask questions with your voice (optional).  
✅ **Privacy First** — No data stored; everything is processed securely.

---

## 🛠 Tech Stack

**Frontend:**
- ⚡ [Vite](https://vitejs.dev/)
- 🎨 [CSS3](https://developer.mozilla.org/en-US/docs/Web/CSS)

**Backend:**
- 🐍 [FastAPI](https://fastapi.tiangolo.com/)
- 🤖 [Groq API](https://groq.com/)
- 📜 [LangChain](https://www.langchain.com/)

**Other:**
- 🔐 `.env` for environment variables
- 🌐 REST API communication

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/maria2469/Youtube-Chatbot-Extension.git

# Go into the project directory
cd Youtube-Chatbot-Extension

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../extension
npm install

# Start the backend
uvicorn main:app --reload

# Start the extension (development build)
npm run dev
