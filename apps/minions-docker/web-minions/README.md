# Advanced Minions Web Interface

This is the advanced web interface for the Minions Docker setup, specifically designed to work with the full Minions protocol (`minions_http_server.py`).

## Features

- **Advanced Task Configuration**: Support for complex task descriptions with document metadata
- **Retrieval Options**: BM25 retrieval support with configurable chunking strategies
- **Advanced Parameters**: Full control over minions protocol parameters:
  - Max jobs per round
  - Number of tasks per round
  - Number of samples per task
  - Chunking functions (by section, page, paragraph, code, function/class)
- **Real-time Status**: Backend health monitoring and configuration display
- **Detailed Metrics**: Comprehensive execution metrics and timing information
- **Modern UI**: Responsive design with collapsible advanced options

## Differences from Basic Interface

This interface is designed for the **full Minions protocol** (`docker-compose.minions.yml`) and differs from the basic interface in several ways:

| Feature | Basic Interface (`web/`) | Advanced Interface (`web-minions/`) |
|---------|-------------------------|-------------------------------------|
| Backend Endpoint | `/run` | `/minions` |
| Request Format | `{"query": "...", "context": [...]}` | `{"task": "...", "doc_metadata": "...", "context": [...]}` |
| Retrieval Support | No | Yes (BM25) |
| Chunking Options | No | Yes (5 strategies) |
| Advanced Parameters | Limited | Full control |
| UI Complexity | Simple | Advanced with collapsible sections |

## API Compatibility

This interface is specifically designed to work with:
- `minions_http_server.py` (full Minions protocol)
- `/minions` endpoint
- Advanced parameter support
- Retrieval and chunking features

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Docker Usage

This interface is automatically built and served when using `docker-compose.minions.yml`:

```bash
# Start the advanced minions setup
docker-compose -f docker-compose.minions.yml up

# Access the interface at http://localhost:8080
```

## Configuration

The interface automatically detects the backend URL and adapts to different deployment scenarios:
- Docker Compose: Uses service name resolution
- Local development: Uses localhost/127.0.0.1
- Automatic port detection and conflict avoidance

## Advanced Features

### Retrieval Configuration
- **BM25 Retrieval**: Keyword-based document retrieval
- **Chunking Strategies**: Multiple ways to split documents
- **Retrieval Model**: Configurable embedding model

### Task Management
- **Auto-generated Logging IDs**: Based on task description
- **Document Metadata**: Specify document type for better processing
- **Context Management**: Support for large document contexts

### Monitoring
- **Real-time Status**: Backend health and configuration monitoring
- **Detailed Metrics**: Token usage, timing, and parameter tracking
- **Error Handling**: Comprehensive error reporting and debugging

## Browser Support

- Modern browsers with ES6+ support
- Chrome 60+, Firefox 55+, Safari 12+, Edge 79+
