# 🎙️ AI Meeting Assistant with RAG-Powered Meeting Intelligence

Transform meeting recordings, YouTube videos, and audio files into actionable insights using AI.

AI Meeting Assistant is an end-to-end meeting intelligence platform that automatically transcribes audio/video, generates summaries, extracts action items, identifies decisions, detects risks, and enables conversational Q&A using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

### 🎧 Audio & Video Processing

* Process YouTube videos directly
* Upload local audio/video files
* Automatic audio extraction and chunking
* Support for large recordings

### 📝 Speech-to-Text Transcription

* OpenAI Whisper integration
* Sarvam AI support for Hinglish conversations
* Multi-language workflow support
* High-quality transcript generation

### 📋 AI-Powered Summarization

* Automatic meeting title generation
* Concise meeting summaries
* Key discussion highlights
* Executive-level meeting overview

### 🔍 Meeting Intelligence Extraction

* ✅ Action Items
* 🔑 Key Decisions
* ❓ Open Questions
* ⚠ Risks & Blockers

### 🧠 RAG-Based Meeting Chat

* Ask questions about the meeting
* Context-aware responses
* Semantic search over transcripts
* Conversational memory powered by vector embeddings

### 📄 Export & Reporting

* Transcript download
* Summary download
* Report generation
* Meeting history storage

### 🎨 Modern Dashboard

* Interactive Streamlit UI
* Real-time pipeline status
* Dark mode interface
* Chat-style experience

---

# 🏗️ Architecture

```text
YouTube URL / Audio / Video
              │
              ▼
      Audio Processing
              │
              ▼
        Transcription
    (Whisper / Sarvam)
              │
              ▼
      Meeting Transcript
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
 Summary  Extraction   RAG
              │        │
              ▼        ▼
 Action Items  ChromaDB
 Decisions         │
 Questions         ▼
 Risks        Meeting Chat
```

---

# 🛠️ Tech Stack

## Frontend

* Streamlit

## LLM & AI

* Mistral AI
* LangChain
* Retrieval-Augmented Generation (RAG)

## Speech Processing

* OpenAI Whisper
* Sarvam AI

## Vector Database

* ChromaDB
* HuggingFace Embeddings

## Data Processing

* Python
* Pandas
* NumPy

## Audio Processing

* FFmpeg
* Pydub
* yt-dlp

---

# 📂 Project Structure

```text
AI-Meeting-Assistant/
│
├── app.py
├── main.py
├── test.py
├── requirements.txt
├── .env.example
│
├── core/
│   ├── transcriber.py
│   ├── summarizer.py
│   ├── extractor.py
│   ├── vector_store.py
│   └── rag_engine.py
│
├── utils/
│   └── audio_processor.py
│
├── reports/
├── uploads/
├── downloads/
└── vector_db/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Meeting-Assistant.git

cd AI-Meeting-Assistant
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
uv pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key

SARVAM_API_KEY=your_sarvam_api_key

WHISPER_MODEL=base
```

---

# ▶️ Run Application

## Streamlit Dashboard

```bash
streamlit run app.py
```

## CLI Version

```bash
python main.py
```

---

# 📸 Screenshots

## Dashboard

Add screenshot here:

```text
screenshots/dashboard.png
```

![Dashboard](screenshots/dashboard.png)

---

## Meeting Summary

Add screenshot here:

```text
screenshots/summary.png
```

![Summary](screenshots/summary.png)

---

## Action Items & Decisions

Add screenshot here:

```text
screenshots/insights.png
```

![Insights](screenshots/insights.png)

---

## RAG Chat Interface

Add screenshot here:

```text
screenshots/chat.png
```

![Chat](screenshots/chat.png)

---

# 💬 Example Questions

Ask your meeting:

```text
What were the key decisions made?

What action items were assigned?

What risks were identified?

Summarize the discussion in 5 points.

Who was responsible for the next steps?

What questions remain unanswered?
```

---

# 📈 Future Improvements

* PDF Report Export
* Multi-Meeting Search
* Speaker Diarization
* Team Collaboration
* Cloud Deployment
* FastAPI Backend
* Next.js Frontend
* Authentication & User Accounts

---

# 🎯 Use Cases

* Business Meetings
* Team Standups
* Client Calls
* Interviews
* Lectures
* Webinars
* Podcasts
* YouTube Content Analysis

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Piyush Joshi

B.Tech Student | AI & Software Development Enthusiast

GitHub: https://github.com/piyushjoshi9351/AI-Meeting-Assistant

LinkedIn: https://www.linkedin.com/in/piyush-joshi-18a152277/

---

⭐ If you found this project useful, consider giving it a star.
