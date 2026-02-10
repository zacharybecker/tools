# PDF to Markdown Converter

A FastAPI-based service that converts PDF files to Markdown using LiteLLM vision models. Designed for integration with OpenWebUI for handling PDF file uploads.

## Features

- **Async Processing**: Handles multiple PDF pages in parallel for fast conversion
- **Concurrent Users**: Built on FastAPI to handle multiple simultaneous requests
- **Retry Logic**: Automatic retry with exponential backoff for rate limits and failures
- **Vision Model Integration**: Uses LiteLLM to call vision-capable models (GPT-4V, Claude, etc.)
- **Robust Error Handling**: Gracefully handles failures, returns partial results when possible
- **Docker Ready**: Containerized for easy deployment

## Architecture

1. **PDF Upload**: Accepts base64-encoded PDF via JSON API
2. **Page Extraction**: Converts each PDF page to a high-quality image (300 DPI)
3. **Parallel Processing**: Sends page images to LiteLLM concurrently (configurable limit)
4. **Markdown Generation**: Vision model converts each page to structured Markdown
5. **Result Aggregation**: Combines all pages into a single Markdown document

## Prerequisites

- Python 3.11+
- LiteLLM instance running and accessible
- Docker (for containerized deployment)

## Installation

### Local Development

```bash
# Clone the repository
cd pdf-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your LiteLLM configuration
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t pdf-to-markdown .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e LITELLM_BASE_URL=http://your-litellm:4000 \
  -e LITELLM_MODEL=gpt-4-vision-preview \
  --name pdf-converter \
  pdf-to-markdown
```

## Configuration

Configure via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_BASE_URL` | LiteLLM API endpoint | *Required* |
| `LITELLM_MODEL` | Vision model to use | *Required* |
| `LITELLM_API_KEY` | API key if required | None |
| `MAX_CONCURRENT_LLM_CALLS` | Max parallel LLM requests | 5 |
| `MAX_PDF_SIZE_MB` | Maximum PDF file size | 50 |
| `MAX_RETRIES` | Retry attempts on failure | 5 |
| `IMAGE_DPI` | Image resolution for conversion | 300 |

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "litellm_configured": true
}
```

### Convert PDF to Markdown

```bash
# Prepare base64-encoded PDF
PDF_BASE64=$(base64 -w 0 document.pdf)

# Send conversion request
curl -X POST http://localhost:8000/convert \
  -H "Content-Type: application/json" \
  -d "{\"pdf_data\": \"$PDF_BASE64\"}"
```

Response:
```json
{
  "markdown": "# Page 1\n\n[content]...\n\n---\n\n# Page 2\n\n[content]...",
  "total_pages": 10,
  "successful_pages": 10,
  "failed_pages": 0,
  "errors": null,
  "processing_time_seconds": 45.2
}
```

### Partial Success Response

If some pages fail, you still get the successful pages:

```json
{
  "markdown": "# Page 1\n\n[content]...\n\n---\n\n# Page 2\n\n[content]...",
  "total_pages": 10,
  "successful_pages": 8,
  "failed_pages": 2,
  "errors": [
    {"page": 3, "error": "Rate limit exceeded"},
    {"page": 7, "error": "Timeout"}
  ],
  "processing_time_seconds": 120.5
}
```

## OpenWebUI Integration

This service is designed to work with OpenWebUI's file processing pipeline. OpenWebUI will:

1. Detect uploaded PDF files
2. Send base64-encoded PDF to this service's `/convert` endpoint
3. Receive Markdown response
4. Display or process the Markdown content

Configure OpenWebUI to use this service as a PDF processing backend.

## Performance Tuning

- **Concurrent LLM Calls**: Increase `MAX_CONCURRENT_LLM_CALLS` if your LiteLLM instance can handle more load
- **Image DPI**: Lower `IMAGE_DPI` (e.g., 150) for faster processing with slightly lower quality
- **Worker Processes**: Run multiple uvicorn workers for higher throughput:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
  ```

## Error Handling

The service implements multiple levels of error handling:

1. **Request Validation**: Invalid base64 or oversized PDFs return 400 errors
2. **PDF Parsing**: Corrupted PDFs return clear error messages
3. **LLM Failures**: Individual page failures don't block entire conversion
4. **Retry Logic**: Automatic retry with exponential backoff for transient failures
5. **Partial Results**: Returns successfully converted pages even if some fail

## Monitoring

- **Health Endpoint**: `/health` for basic service status
- **Logs**: Structured logging with INFO level for operations, ERROR for failures
- **Metrics**: Processing time included in response for performance monitoring

## Development

Run tests (if implemented):
```bash
pytest
```

Run with auto-reload for development:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### "Invalid or corrupted PDF"
- Ensure the PDF is valid and not encrypted
- Check that base64 encoding is correct

### "Rate limit exceeded"
- Reduce `MAX_CONCURRENT_LLM_CALLS`
- Check LiteLLM rate limits

### "Timeout errors"
- Large PDFs may take time; ensure timeout settings are appropriate
- Consider processing very large PDFs in smaller batches

### "Connection refused to LiteLLM"
- Verify `LITELLM_BASE_URL` is correct
- Ensure LiteLLM service is running and accessible

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.
