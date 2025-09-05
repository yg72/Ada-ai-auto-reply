# Tuilink Project - AI Auto Reply

An intelligent system for generating contextually appropriate replies in professional conversations, particularly focused on job referral scenarios.

## Overview

This project implements an automated reply generation system that:

1. Analyzes conversation context
2. Classifies conversation categories
3. Suggests relevant topics for replies
4. Generates professional and contextually appropriate responses

## Project Structure

```
.
├── input/                  # Input data directory
│   ├── categories.json     # Category definitions
│   └── convo_2454_rows.xlsx # Conversation dataset
├── models/                 # Core data models
├── nodes/                  # Processing nodes
├── output/                 # Generated outputs
├── utils/                  # Utility functions
└── run.ipynb              # Main execution notebook
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project uses environment variables for configuration. Create a `.env` file with the following necessary credentials and settings:

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

## Build for Deployment

Use the build script to create a self-contained `dist/` folder ready for packaging and deployment.

1. Ensure your `.env` is present if you want it copied into the build output.

2. Run the build script:

```bash
chmod +x ./build.sh
./build.sh
```

3. The script will create `./dist` containing:

- `ai_auto_reply.py`
- `requirements.txt` (copied from `requirements-compact.txt`)
- `models/`, `nodes/`, `utils/`, `input/`, `output/` (project files only)
- `__init__.py` files for Python package directories (`models/`, `nodes/`, `utils/`)

4. Deployment notes for your infra repo:

- Set `AI_AUTO_REPLY_LAMBDA_SOURCE_PATH` to be the path to the `dist` folder of this repo, e.g. `../../tuilink-project-ai-auto-reply/dist/`

- Set `AI_AUTO_REPLY_LAMBDA_HANDLER` to be `ai_auto_reply.handler`
