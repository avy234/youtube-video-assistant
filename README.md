# YouTube Video Assistant

An AI-powered assistant that can answer questions about any YouTube video using its transcript. Built with LangChain and OpenAI's GPT-3.5 Turbo.

## How It Works

1. **Transcript Extraction** — Downloads the transcript from a YouTube video using `YoutubeLoader`
2. **Text Chunking** — Splits the transcript into manageable chunks using `RecursiveCharacterTextSplitter`
3. **Vector Embedding** — Converts chunks into vector embeddings using OpenAI's embedding model and stores them in a FAISS vector database
4. **Similarity Search** — When you ask a question, it finds the most relevant transcript chunks
5. **AI Response** — Sends the relevant chunks + your question to GPT-3.5 Turbo to generate an accurate answer

## Tech Stack

- **LangChain** — Framework for building LLM applications
- **OpenAI GPT-3.5 Turbo** — Language model for generating responses
- **FAISS** — Vector database for fast similarity search
- **OpenAI Embeddings** — Text-to-vector conversion

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/avy234/youtube-video-assistant.git
cd youtube-video-assistant
```

### 2. Create a virtual environment

```bash
python -m venv env
```

**Activate it:**

- Windows PowerShell: `.\env\Scripts\Activate.ps1`
- macOS/Linux: `source env/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example environment file and add your OpenAI API key:

```bash
cp example.env .env
```

Then edit `.env` and replace `your-key-here` with your actual [OpenAI API key](https://platform.openai.com/api-keys).

### 5. Run the assistant

Run the script directly:

```bash
python youtube_chat.py
```

Or open it in VS Code and use the Interactive Window (click "Run Cell" above each `# %%` marker).

## Example Queries

The script includes 3 demo videos with sample queries:

| Video Topic | Query |
|---|---|
| Best Dark Type Pokemon | "What is the number 10 in the list of the best dark type pokemon?" |
| Investment Methods | "What is the investment method that consistently deliver returns over time?" |
| Chess: Caro-Kann Opening | "What are the pieces objectives when you play the Caro-Kann opening as black?" |

## Usage with Your Own Videos

Replace the video URL and query with any YouTube video that has a transcript:

```python
video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
db = create_db_from_youtube_video_url(video_url)

query = "Your question about the video"
response, docs = get_response_from_query(db, query)
print(response)
```

## Acknowledgments

Inspired by [langchain-experiments](https://github.com/engchina/langchain-experiments) by engchina.
