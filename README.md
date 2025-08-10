# Data Analyst Agent

A Python-based agentic data analyst that can:
- Read and extract data from PDF files, images, and web pages
- Perform DuckDB SQL operations on structured data
- Generate plots and visualizations (including regression lines)
- Accept questions and data via a simple Flask API

## Features
- **Flexible Data Ingestion:** Accepts PDF, image, CSV, and web page data as attachments or URLs.
- **LLM Orchestration:** Uses OpenAI function calling to decide which tools to use for data extraction, analysis, and visualization.
- **DuckDB Integration:** Run SQL queries on structured data, including aggregations and advanced analytics.
- **Visualization:** Generates scatterplots and other visualizations, returning results as base64-encoded images.
- **API-Driven:** Interact with the agent via an API endpoint (`POST /24f1002354-submit-question`) using `curl` or any HTTP client.

## System Requirements
- The `libcairo2` package must be installed on your system. This is required by the `cairosvg` Python package for SVG-to-PNG conversion.
  - On Ubuntu/Debian: `sudo apt install libcairo2`

## Usage

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Set up your API key
Copy `.env.example` to `.env` and add your OpenAI API key:
```
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run the API server
```
python Api.py
```

### 4. Ask a question with data attachments
Prepare a `questions.txt` file with your analysis question(s). Optionally, include data files (e.g., CSV, PDF, PNG).

Example:
```
curl "http://localhost:9004/24f1002354-submit-question" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

The agent will:
- Save all attachments to a temp directory
- Note their file paths in the question string
- Use LLM-driven tools to extract, analyze, and visualize data
- Return a JSON answer (including base64-encoded images if needed)

## Supported Data Types
- PDF files (tables, text extraction)
- Images (charts, plots, screenshots)
- Web pages (HTML tables, unstructured data)
- CSV and Parquet files

## License
MIT
