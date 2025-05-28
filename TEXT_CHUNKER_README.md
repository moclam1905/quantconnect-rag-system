# Text Chunker Documentation

## Overview

Text Chunker là component chịu trách nhiệm chia nhỏ các sections đã parse thành các chunks phù hợp cho embedding và vector storage.

## Components

### 1. `text_chunker.py`
- **TextChunker**: Basic chunker với size-based splitting
- **AdvancedTextChunker**: Smart chunker với paragraph grouping

### 2. `test_text_chunker.py`
- Test suite để validate chunking với real data
- Compare different chunking strategies

### 3. `chunking_pipeline.py`
- Production pipeline để process parsed files
- Batch processing support

## Usage

### Quick Start

```python
# Chunk a single parsed file
from src.data_processing.chunking_pipeline import chunk_single_file

stats = chunk_single_file(
    Path("data/processed/Quantconnect-Writing-Algorithms_parsed.json"),
    chunker_type="advanced"
)
```

### Batch Process All Files

```bash
# From command line
python src/data_processing/chunking_pipeline.py

# Or in code
from src.data_processing.chunking_pipeline import chunk_all_parsed_files
results = chunk_all_parsed_files(chunker_type="advanced")
```

### Custom Configuration

```python
from src.data_processing.chunk_models import ChunkingConfig
from src.data_processing.text_chunker import TextChunker

# Create custom config
config = ChunkingConfig(
    max_chunk_size=800,      # Smaller chunks
    chunk_overlap=100,       # Less overlap
    ensure_complete_sentences=True
)

# Use with chunker
chunker = TextChunker(config)
chunks = chunker.chunk_text(text, source_file, section)
```

## Chunking Strategies

### Basic Chunker (TextChunker)
- Splits by character count
- Respects word boundaries
- Optional sentence boundary preservation
- Configurable overlap

**Best for**: General text content, consistent chunk sizes

### Advanced Chunker (AdvancedTextChunker)
- Groups paragraphs intelligently
- Preserves headings
- Better context preservation
- Smart overlap based on paragraph boundaries

**Best for**: Documentation, tutorials, structured content

## Configuration Options

Key parameters in `ChunkingConfig`:

- `max_chunk_size`: Maximum characters per chunk (default: 1000)
- `min_chunk_size`: Minimum characters per chunk (default: 100)
- `chunk_overlap`: Character overlap between chunks (default: 200)
- `ensure_complete_sentences`: Don't break sentences (default: True)
- `include_section_header_in_chunks`: Add section info to chunks (default: True)

## Workflow Integration

### Complete Pipeline

```bash
# 1. Parse HTML files
python src/data_processing/batch_process_documents.py
# Choose option 1

# 2. Chunk parsed files
python src/data_processing/chunking_pipeline.py
# This will process all parsed files from step 1

# 3. Check output
ls data/processed/batch_*/
# You'll see:
# - *_parsed.json    (parsed sections)
# - *_rag.json       (formatted for RAG)
# - *_chunks.json    (chunked content)
```

### Output Format

Chunks JSON file contains:
```json
{
  "source_file": "Quantconnect-Writing-Algorithms_parsed.json",
  "chunking_config": {
    "chunker_type": "advanced",
    "timestamp": "2024-01-01T10:00:00"
  },
  "statistics": {
    "total_sections": 50,
    "sections_processed": 45,
    "total_chunks": 150,
    "total_chars": 120000
  },
  "chunks": [
    {
      "chunk_id": "abc123...",
      "content": "Chunk text content...",
      "metadata": {
        "source_file": "Quantconnect-Writing-Algorithms.html",
        "section_id": "1.2.3",
        "section_title": "Algorithm Basics",
        "chunk_index": 0,
        "chunk_type": "text"
      }
    }
  ]
}
```

## Testing

### Test with Sample Data

```bash
python src/data_processing/test_text_chunker.py
# Interactive menu to test different configurations
```

### Compare Chunkers

```python
from src.data_processing.test_text_chunker import compare_chunkers
compare_chunkers(Path("path/to/parsed.json"))
```

## Best Practices

1. **Choose appropriate chunk size**:
   - Embedding models: 300-500 chars
   - Q&A systems: 800-1200 chars
   - Code examples: 1500-2000 chars

2. **Configure overlap wisely**:
   - More overlap = better context preservation
   - Less overlap = more efficient storage
   - Typical: 10-20% of chunk size

3. **Monitor chunk distribution**:
   - Avoid too many very small chunks
   - Check if chunks are semantically complete

4. **Section-specific configs**:
   - Use larger chunks for code sections
   - Use smaller chunks for API references
   - Adjust based on content type

## Next Steps

After chunking:
1. Generate embeddings for each chunk
2. Store in vector database
3. Implement retrieval logic
4. Test with queries

## Troubleshooting

**Issue**: Chunks too small/large
- Adjust `max_chunk_size` in config
- Check if content has unusual formatting

**Issue**: Broken code blocks
- Use larger `max_chunk_size` for code-heavy sections
- Enable `preserve_small_code_blocks` in config

**Issue**: Lost context between chunks
- Increase `chunk_overlap`
- Use AdvancedTextChunker instead of basic

**Issue**: Memory usage with large files
- Process files in batches
- Use streaming approach for very large sections