# TruLens RAG Evaluation System

A comprehensive RAG (Retrieval-Augmented Generation) evaluation system that combines document processing with visual content extraction capabilities, including OCR and image understanding. Built with TruLens for evaluation metrics and LlamaIndex for document processing.

## Key Features

- **Document Processing**: Load and process PDFs, images, and text files
- **Visual Content Extraction**: Extract text from diagrams, charts, and images using OCR
- **QA Generation**: Automatically generate question-answer pairs from documents
- **Comprehensive Evaluation**: Answer Relevance, Context Relevance, Groundedness, Answer Correctness
- **Interactive Dashboard**: Real-time evaluation results visualization
- **Modular Architecture**: Clean, maintainable, and extensible codebase

## Acknowledgments

This project builds upon and enhances the original [TruLens Evaluation Demo](https://github.com/JohannesJolkkonen/funktio-ai-samples/tree/main/trulens-eval-demo) by JohannesJolkkonen. 

**Key improvements include:**
- Enhanced document processing with visual content extraction
- Advanced OCR and vision model integration
- Automated QA pair generation from documents
- Improved modular architecture and code quality
- Professional, production-ready implementation

The original demo provided the foundation for TruLens integration, while this version adds enterprise-grade features and maintainability.

## Project Structure

```
TruLensDemo/
├── Core Modules
│   ├── enhanced_data_prep.py      # Document loading with image support
│   ├── visual_content_processor.py # OCR and visual content extraction
│   ├── indexing.py                 # Document indexing and retrieval
│   ├── evaluation.py               # TruLens metrics and evaluation logic
│   └── generate.py                 # QA pair generation from documents
│
├── Main Applications
│   ├── main.py                     # Complete workflow orchestration
│   └── dashboard.py                # Dashboard launcher
│
├── Configuration
│   ├── .env                        # Environment variables
│   ├── requirements.txt            # Python dependencies
│   └── default.sqlite             # TruLens evaluation database
│
└── Data
    ├── test_documents/             # Place your documents here
    └── your_documents/             # Alternative document location
```

## Installation & Setup

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install poppler  # For PDF to image conversion
```

### 2. Environment Configuration

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Document Setup

Place your documents in the `test_documents/` folder:
```bash
mkdir test_documents
# Copy your PDFs, images, or text files here
```

## Usage

### Quick Start

1. **Run the complete workflow**:
```bash
python main.py
```

2. **Launch dashboard to view results**:
```bash
python dashboard.py
```

### Step-by-Step Workflow

1. **Load Documents**: The system loads your documents using enhanced PDF processing
2. **Build Index**: Creates searchable vector indexes from your documents
3. **Generate QA Pairs**: Automatically generates question-answer pairs for evaluation
4. **Run Evaluation**: Evaluates RAG responses using 4 key metrics
5. **View Results**: Launch the dashboard to see detailed evaluation results

### Customization

- **Document Location**: Update `input_files` in `main.py` to point to your documents
- **QA Generation**: Modify `generate.py` to adjust question generation parameters
- **Evaluation Metrics**: Customize metrics in `evaluation.py`

## Evaluation Metrics

The system evaluates RAG performance using four key metrics:

1. **Answer Relevance**: How relevant is the generated answer to the question?
2. **Context Relevance**: How relevant are the retrieved document chunks?
3. **Groundedness**: How well is the answer supported by the source documents?
4. **Answer Correctness**: How accurate is the answer compared to ground truth?

## Supported File Types

- **PDFs**: Text extraction with image processing
- **Images**: OCR text extraction and vision model analysis
- **Text Files**: Direct loading and processing
- **Documents with Charts/Diagrams**: Visual content understanding

## Dependencies

- **Core**: LlamaIndex, TruLens, OpenAI
- **PDF Processing**: PyMuPDF, PyPDF, pdf2image
- **Vision**: EasyOCR, Tesseract, OpenAI Vision API
- **Data**: Pandas, NumPy

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure your API key is set in the `.env` file
2. **PDF Processing**: Install system dependencies like poppler for PDF conversion
3. **Vision Processing**: Install OCR dependencies for image text extraction
4. **Dashboard Access**: Check that the TruLens database is properly initialized

### Performance Tips

- Use smaller documents for faster testing
- Adjust the number of QA pairs generated based on your needs
- Monitor API usage when processing large documents

## Contributing

This is a modular system designed for easy extension:

- Add new document loaders in `enhanced_data_prep.py`
- Implement new evaluation metrics in `evaluation.py`
- Extend visual processing in `visual_content_processor.py`

## License

This project is open source and available under the MIT License. 