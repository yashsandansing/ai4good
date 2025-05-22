# Legal Document Analyzer — Built for the AI4Good 2025 competition at UB

A FastAPI application that simplifies and analyzes legal documents using LLMs. Built for the AI4Good 2025 competition.

## Features

* Upload legal documents (`.pdf`, `.docx`, `.txt`)
* Generate a summary
* Rate document complexity (1–10)
* Detect risky clauses (red flags)
* Extract financial figures and deadlines
* Flag vague or ambiguous terms (loopholes)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashsandansing/ai4good.git
cd ai4good
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000`

## API

### POST `/process-legal-doc/`

**Request:** Multipart file upload
**Response:**

```json
{
  "summary": "...",
  "complexity_rating": 6,
  "red_flag_detection": ["..."],
  "figures_extraction": ["..."],
  "loopholes": ["..."]
}
```

## Tech Stack

* FastAPI
* llama\_index
* LangChain
* Pydantic
