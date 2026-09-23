# Running locally

Step-by-step setup for running the project from a clean machine. The slow,
network-dependent steps (dependency install, model downloads, corpus ingestion)
are one-time — once done, the server starts instantly.

## Prerequisites

Install these once per machine:

- **Python 3.11+** — verify with `python --version`.
- **Ollama** — https://ollama.com — then pull the default model:
  ```powershell
  ollama pull llama3.2:3b
  ```
- **Git** (only needed to clone the repository).
- For editor use: **VS Code** with the **Python** extension.

## 1. Get the project

Clone it (or open the existing folder):

```powershell
git clone https://github.com/<your-github-username>/<your-repository>.git
cd <your-repository>
```

In VS Code: `File -> Open Folder...` and select the project folder.

## 2. Create the environment and install

From the project root (VS Code terminal: `Ctrl + ~`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The install includes PyTorch, so the first run takes a few minutes.

In VS Code, select the interpreter so the editor uses this environment:
`Ctrl + Shift + P` -> **Python: Select Interpreter** -> choose `.\.venv\`.

> If you rename or move the project folder, recreate the virtual environment
> (`.venv`) rather than reusing the old one.

## 3. Build the document index

Make sure Ollama is running, then download and ingest the demo corpus:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_corpus.py
```

This downloads the papers and the embedding model and writes the vector store
to `.chroma/`. It runs once; subsequent starts reuse the stored index.

## 4. Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8080
```

Open the interactive docs at **http://localhost:8080/docs**. Use
`POST /query` -> *Try it out* -> *Execute* to ask a question. Stop the server
with `Ctrl + C`.

> Port note: if `8080` is in use, pick another free port. On some Windows
> machines port `8000` falls in a reserved range and fails to bind — `8080`
> avoids it.

## 5. Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # unit + control-flow tests
.\.venv\Scripts\python.exe -m eval.run_eval      # metrics table -> eval/results.md
```

## Troubleshooting

- **Queries hang or error** — the Ollama app isn't running, or the model isn't
  pulled (`ollama pull llama3.2:3b`).
- **`running scripts is disabled` in PowerShell** — skip virtual-environment
  activation and call `.\.venv\Scripts\python.exe ...` directly, as shown above.
- **Empty answers / `doc_count` is 0 at `/health`** — run the ingestion step
  (section 3).
