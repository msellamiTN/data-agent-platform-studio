# Data Agent Platform Studio

A Google Colab-first prototype for two data-focused AI agents:
- **Data Engineering Agent** — profile, clean, validate, and prepare CSV data for loading.
- **Data Analysis Agent** — translate business questions into SQL, execute read-only queries, and summarize results.

## Target runtime

- Google Colab with a GPU runtime (T4 when available).
- Gradio web UI with separate UX for both scenarios.
- CrewAI for agent/task/tool orchestration.
- Hugging Face `Qwen/Qwen2.5-1.5B-Instruct` as a compact local model.

The application is designed for a lab/demo workflow. The model proposes and explains; Python tools perform bounded data operations. SQL is read-only and validated before execution.

See the implementation branch for setup and Colab instructions.
