# QuantConnect RAG System - Testing Workflow & Troubleshooting Guide

## 🚀 Quick Start Testing

### 1. Setup Environment
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw_html
mkdir -p data/processed
mkdir -p logs
```

### 2. Prepare QuantConnect HTML Files
Đặt các file HTML từ QuantConnect vào `data/raw_html/`:
- `Quantconnect-Lean-Engine.html`
- `Quantconnect-Writing-Algorithms.html` 
- `Quantconnect-Lean-Cli.html`
- `Quantconnect-Research-Environment.html`

### 3. Run Quick Validation
```python
python src/data_processing/code_validation_guide.py
# Chọn option 1: Quick validation check
```

## 🔍 Comprehensive Testing Workflow

### Step 1: Validate HTML Parser
```python
# Test document structure
from src.data_processing.parser_utils import count_documents_in_html
from pathlib import Path

html_file = Path("data/raw_html/Quantconnect-Lean-Engine.html")
doc_count = count_documents_in_html(html_file)
print(f"Documents found: {doc_count}")
# Expected: 2 (document 0 is usually cover page, document 1 is main content)
```

### Step 2: Test HTML Parsing
```python
from src.data_processing.html_parser import QuantConnectHTMLParser

# Parse main document (index 1)
parser = QuantConnectHTMLParser(html_file)
sections = parser.parse(target_document_index=1)

print(f"Sections extracted: {len(sections)}")
print("Sample sections:")
for i, section in enumerate(sections[:3]):
    print(f"  {i+1}. {section.id} - {section.title} (Level {section.level})")
    print(f"     Content length: {len(section.content or '')}")
    print(f"     Code blocks: {len(section.code_blocks)}")
    print(f"     Tables: {len(section.tables)}")
```

### Step 3: Test Chunking Strategies
```python
from src.data_processing.chunk_models import ChunkingPresets
from src.data_processing.hybrid_chunker import HybridChunker

# Test with hybrid chunker
config = ChunkingPresets.for_documentation()
chunker = HybridChunker(config)

# Test single section
test_section = sections[0]
chunks = chunker.chunk_section(test_section, html_file.name, 1)

print(f"Chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks[:2]):
    print(f"\nChunk {i+1}:")
    print(f"  Size: {chunk.char_count} chars")
    print(f"  Type: {chunk.metadata.chunk_type.value}")
    print(f"  Preview: {chunk.content[:100]}...")
```

### Step 4: Run Full Validation Suite
```python
from src.data_processing.code_validation_guide import CodeValidator

validator = CodeValidator()
results = validator.run_full_validation()
validator.save_validation_report()
```

## 🐛 Common Issues & Troubleshooting

### Issue 1: No Sections Extracted
**Symptoms:**
- `sections = []` after parsing
- Warning: "No <section> elements found"

**Possible Causes:**
1. Wrong document index (trying to parse cover page)
2. HTML structure different than expected
3. File encoding issues

**Solutions:**
```python
# Check document count first
doc_count = count_documents_in_html(html_file)
print(f"Available documents: {doc_count}")

# Try different document indices
for i in range(doc_count):
    try:
        sections = parser.parse(target_document_index=i)
        print(f"Document {i}: {len(sections)} sections")
    except Exception as e:
        print(f"Document {i}: Error - {e}")
```

### Issue 2: Code Blocks Not Detected
**Symptoms:**
- `section.code_blocks = []` for sections that should have code
- Code content appears in regular text

**Debug Steps:**
```python
# Check raw HTML structure
with open(html_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Look for code patterns
import re
code_patterns = [
    r'<pre[^>]*>.*?</pre>',
    r'<code[^>]*>.*?</code>',
    r'```.*?```'
]

for pattern in code_patterns:
    matches = re.findall(pattern, html_content, re.DOTALL)
    print(f"Pattern '{pattern}': {len(matches)} matches")
```

### Issue 3: Section Hierarchy Issues
**Symptoms:**
- All sections at level 1
- Missing parent-child relationships
- Incorrect breadcrumbs

**Debug Steps:**
```python
# Check ToC structure
parser = QuantConnectHTMLParser(html_file)
parser._load_html(target_document_index=1)
parser._parse_table_of_contents()

print("ToC Structure:")
for section_id, info in list(parser.toc_structure.items())[:5]:
    print(f"  {section_id}: Level {info['level']} - {info['title']}")
```

### Issue 4: Chunking Produces Empty/Invalid Chunks
**Symptoms:**
- Chunks with no content
- Chunks with only headers
- Extremely small or large chunks

**Debug Steps:**
```python
# Check chunk quality
for i, chunk in enumerate(chunks):
    if chunk.char_count < 50:
        print(f"Small chunk {i}: {chunk.char_count} chars")
        print(f"Content: {repr(chunk.content[:100])}")
    
    if chunk.char_count > 2000:
        print(f"Large chunk {i}: {chunk.char_count} chars")
        print(f"Type: {chunk.metadata.chunk_type.value}")
```

### Issue 5: Memory Issues with Large Files
**Symptoms:**
- Python crashes or hangs
- Memory usage keeps increasing

**Solutions:**
```python
# Process files one by one
import gc

for html_file in html_files:
    try:
        # Process file
        parser = QuantConnectHTMLParser(html_file)
        sections = parser.parse(target_document_index=1)
        
        # Process in batches
        for i in range(0, len(sections), 5):  # Process 5 sections at a time
            batch = sections[i:i+5]
            # Process batch...
            
        # Clean up
        del parser, sections
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {html_file}: {e}")
```

## 🔧 Performance Optimization Tips

### 1. HTML Parser Optimization
```python
# Use target_document_index to avoid parsing unnecessary content
parser = QuantConnectHTMLParser(html_file)
sections = parser.parse(target_document_index=1)  # Skip cover page

# Clean HTML more aggressively for large files
parser._clean_html()  # Remove unnecessary elements early
```

### 2. Chunking Optimization
```python
# Use appropriate chunk sizes for your use case
config = ChunkingConfig(
    max_chunk_size=1000,    # Smaller for better retrieval
    chunk_overlap=150,      # Reasonable overlap
    min_chunk_size=200      # Avoid tiny chunks
)

# Choose chunker based on content type
if has_lots_of_code:
    chunker = CodeAwareChunker(config)
elif well_structured:
    chunker = SectionBasedChunker(config)
else:
    chunker = HybridChunker(config)  # Safe default
```

### 3. Memory Management
```python
# Process large files in streaming fashion
def process_large_file(html_file):
    parser = QuantConnectHTMLParser(html_file)
    
    # Parse sections lazily if possible
    sections = parser.parse(target_document_index=1)
    
    # Process sections in batches
    batch_size = 10
    for i in range(0, len(sections), batch_size):
        batch = sections[i:i+batch_size]
        yield from process_section_batch(batch)
```

## 📊 Quality Validation Checklist

### HTML Parser Quality
- [ ] Correct number of documents detected
- [ ] All sections have valid IDs and titles
- [ ] Section hierarchy properly built (parent-child relationships)
- [ ] Code blocks extracted with correct language detection
- [ ] Tables extracted with headers and data
- [ ] Breadcrumbs populated correctly

### Chunking Quality
- [ ] No empty chunks produced
- [ ] Chunk sizes within reasonable range (100-2000 chars)
- [ ] Proper overlap between consecutive chunks
- [ ] Complete sentences/code blocks (no mid-sentence cuts)
- [ ] Metadata properly populated
- [ ] Chunk types correctly identified

### Performance Quality
- [ ] Processing time reasonable (< 30s per file)
- [ ] Memory usage stable (no memory leaks)
- [ ] Error handling graceful
- [ ] Large files processed without crashes

## 🧪 Test Data Requirements

### Ideal Test Files Should Have:
1. **Multiple document structure** (cover page + main content)
2. **Hierarchical sections** (levels 1, 2, 3+)
3. **Mixed content types**:
   - Pure text sections
   - Code-heavy sections
   - Sections with tables
   - Mixed text + code sections
4. **Different programming languages** (Python, C#)
5. **Various section sizes** (small, medium, large)

### Test File Structure Validation:
```python
def validate_test_file(html_file):
    """Validate that test file has good structure for testing"""
    
    # Check document count
    doc_count = count_documents_in_html(html_file)
    assert doc_count >= 2, "Need at least 2 documents"
    
    # Parse main document
    parser = QuantConnectHTMLParser(html_file)
    sections = parser.parse(target_document_index=1)
    
    # Check variety
    levels = set(s.level for s in sections)
    assert len(levels) > 1, "Need hierarchical structure"
    
    code_sections = [s for s in sections if s.code_blocks]
    assert len(code_sections) > 0, "Need sections with code"
    
    languages = set()
    for s in sections:
        for cb in s.code_blocks:
            languages.add(cb.language)
    assert len(languages) > 0, "Need code in different languages"
    
    print(f"✅ Test file validation passed:")
    print(f"   - {doc_count} documents")
    print(f"   - {len(sections)} sections")
    print(f"   - {len(levels)} hierarchy levels")
    print(f"   - {len(code_sections)} sections with code")
    print(f"   - Languages: {', '.join(languages)}")
```

## 🎯 Recommended Testing Sequence

1. **Quick Setup Check** (5 minutes)
   - Run quick validation
   - Check file availability
   - Test basic parsing

2. **Component Testing** (15 minutes)
   - Test HTML parser thoroughly
   - Test each chunker individually
   - Validate chunk quality

3. **Integration Testing** (10 minutes)
   - Test full pipeline
   - Test multiple files
   - Check memory usage

4. **Performance Testing** (10 minutes)
   - Benchmark different chunkers
   - Optimize parameters
   - Validate scalability

5. **Quality Assurance** (10 minutes)
   - Run full validation suite
   - Review error logs
   - Generate test report

**Total Time: ~50 minutes for comprehensive testing**

## 📋 Pre-Production Checklist

Before moving to production:

- [ ] All validation tests pass
- [ ] Performance benchmarks acceptable
- [ ] Memory usage within limits
- [ ] Error handling tested
- [ ] Configuration parameters optimized
- [ ] Test report generated and reviewed
- [ ] Code review completed
- [ ] Documentation updated

---

**Next Steps:** Once validation passes, proceed to Vector Database setup and RAG Pipeline implementation.