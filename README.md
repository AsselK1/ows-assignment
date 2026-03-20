# OWS Assignment — Procurement AI Agent

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**

   ```bash
   uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
   ```

3. **In a separate terminal, run the chat client:**

   ```bash
   python chat.py
   ```

   The chat connects to `http://localhost:8000` by default. To use a different URL:

   ```bash
   python chat.py --url http://your-server:8000
   ```

> **Note:** The very first prompt may take a bit of time to process while the model and vector store initialize. Subsequent queries will be faster.
