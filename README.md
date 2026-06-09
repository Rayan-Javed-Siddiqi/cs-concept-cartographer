# CS Concept Cartographer

An AI-powered study tool that turns any computer science topic into an interactive knowledge graph. Built with **Streamlit**, **Google Gemini**, and **vis.js**.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.55-red)

## Features

- **Map** — Generate concept graphs with typed nodes (prerequisite, related, advanced, application)
- **Compare** — View two topics side by side
- **Quiz** — Hide labels and test yourself on node descriptions
- **History** — SQLite-backed cache per profile with search, rename, and export
- **Annotations** — Add personal notes to any node

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/Rayan-Javed-Siddiqi/cs-concept-cartographer.git
cd cs-concept-cartographer
python -m venv ai_env

# Windows
ai_env\Scripts\activate

# macOS / Linux
source ai_env/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=...
```

Or enter the key in the app sidebar at runtime.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. Add `GEMINI_API_KEY` under **Secrets**.

## Tech stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-3-flash-preview`) |
| Graph viz | vis.js (embedded HTML component) |
| Storage | SQLite (`maps.db`) |

## Project structure

```
app.py              # Application entry point
requirements.txt    # Python dependencies
.env.example        # Environment variable template
maps.db             # Created at runtime (gitignored)
```

## License

MIT
