#!/usr/bin/env python3

# Visual content processing with OCR and vision model integration

import os
import io
import base64
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
from PIL import Image
import numpy as np

# OCR and Vision imports
try:
    import easyocr
    import pytesseract
    from pdf2image import convert_from_path
    VISUAL_DEPS_AVAILABLE = True
except ImportError:
    VISUAL_DEPS_AVAILABLE = False

# OpenAI for vision
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from llama_index.core import Document
from enhanced_data_prep import load_documents_with_images

class VisualContentProcessor:
    # Advanced visual content processor for PDFs with diagrams and charts
    
    def __init__(self, use_openai_vision=True, use_easyocr=True, use_tesseract=False):
        self.use_openai_vision = use_openai_vision and OPENAI_AVAILABLE
        self.use_easyocr = use_easyocr and VISUAL_DEPS_AVAILABLE
        self.use_tesseract = use_tesseract and VISUAL_DEPS_AVAILABLE
        
        # Initialize OCR readers
        if self.use_easyocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                print("EasyOCR initialized")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.use_easyocr = False
        
        # Initialize OpenAI client
        if self.use_openai_vision:
            try:
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                print("OpenAI Vision initialized")
            except Exception as e:
                print(f"OpenAI Vision initialization failed: {e}")
                self.use_openai_vision = False
    
    def process_pdf_with_visual_content(self, pdf_path: Path, output_dir: Optional[Path] = None) -> List[Document]:
        # Process PDF with advanced visual content extraction
        print(f"Processing PDF with visual content: {pdf_path.name}")
        
        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_visual_content"
        output_dir.mkdir(exist_ok=True)
        
        # Load documents with basic processing
        documents = load_documents_with_images(input_files=[str(pdf_path)], extract_images=True)
        print(f"Loaded {len(documents)} base documents")
        
        # Convert PDF pages to images for visual processing
        try:
            page_images = convert_from_path(str(pdf_path), dpi=300)
            print(f"Converted {len(page_images)} pages to images")
        except Exception as e:
            print(f"Failed to convert PDF to images: {e}")
            return documents
        
        # Process each page image
        enhanced_documents = []
        for i, (doc, page_image) in enumerate(zip(documents, page_images)):
            print(f"Processing page {i+1}/{len(documents)}")
            
            # Save page image
            page_image_path = output_dir / f"page_{i+1}.png"
            page_image.save(page_image_path)
            
            # Extract visual content
            visual_content = self.extract_visual_content(page_image, page_image_path)
            
            # Enhance document with visual content
            enhanced_text = self.enhance_document_with_visual_content(doc.text, visual_content)
            
            # Create enhanced document
            enhanced_doc = Document(
                text=enhanced_text,
                metadata={
                    **doc.metadata,
                    "visual_content": visual_content,
                    "page_image_path": str(page_image_path)
                }
            )
            enhanced_documents.append(enhanced_doc)
        
        print(f"Enhanced {len(enhanced_documents)} documents with visual content")
        return enhanced_documents
    
    def extract_visual_content(self, image: Image.Image, image_path: Path) -> Dict[str, Any]:
        # Extract visual content from a single image
        visual_content = {
            "ocr_text": "",
            "image_description": "",
            "charts_detected": False,
            "diagrams_detected": False,
            "tables_detected": False
        }
        
        # OCR text extraction
        if self.use_easyocr:
            try:
                ocr_results = self.easyocr_reader.readtext(np.array(image))
                ocr_text = " ".join([result[1] for result in ocr_results])
                visual_content["ocr_text"] = ocr_text
                print(f"Extracted {len(ocr_text)} characters via OCR")
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        # Vision model description
        if self.use_openai_vision:
            try:
                description = self.describe_image_with_vision_model(image)
                visual_content["image_description"] = description
                
                # Detect content types
                description_lower = description.lower()
                visual_content["charts_detected"] = any(word in description_lower for word in ["chart", "graph", "plot", "bar", "line"])
                visual_content["diagrams_detected"] = any(word in description_lower for word in ["diagram", "flowchart", "schematic", "figure"])
                visual_content["tables_detected"] = any(word in description_lower for word in ["table", "data", "rows", "columns"])
                
                print(f"Generated image description: {description[:100]}...")
            except Exception as e:
                print(f"Vision model failed: {e}")
        
        return visual_content
    
    def describe_image_with_vision_model(self, image: Image.Image) -> str:
        # Use OpenAI's vision model to describe an image
        if not self.use_openai_vision:
            return ""
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on any charts, diagrams, tables, or visual data. If there are any scientific figures, explain what they show. Be specific about data trends, relationships, and key insights."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI Vision API error: {e}")
            return ""
    
    def enhance_document_with_visual_content(self, original_text: str, visual_content: Dict[str, Any]) -> str:
        # Enhance document text with extracted visual content
        enhanced_text = original_text
        
        # Add OCR text if available
        if visual_content.get("ocr_text"):
            enhanced_text += f"\n\n[OCR EXTRACTED TEXT]\n{visual_content['ocr_text']}"
        
        # Add image description if available
        if visual_content.get("image_description"):
            enhanced_text += f"\n\n[VISUAL CONTENT DESCRIPTION]\n{visual_content['image_description']}"
        
        # Add content type indicators
        content_types = []
        if visual_content.get("charts_detected"):
            content_types.append("charts/graphs")
        if visual_content.get("diagrams_detected"):
            content_types.append("diagrams/figures")
        if visual_content.get("tables_detected"):
            content_types.append("tables/data")
        
        if content_types:
            enhanced_text += f"\n\n[VISUAL CONTENT TYPES]\nThis page contains: {', '.join(content_types)}"
        
        return enhanced_text

def process_pdf_with_visual_understanding(pdf_path: str, output_dir: Optional[str] = None) -> List[Document]:
    # Convenience function to process PDF with visual understanding
    processor = VisualContentProcessor()
    return processor.process_pdf_with_visual_content(Path(pdf_path), Path(output_dir) if output_dir else None)

def create_sample_pdf_with_chart():
    # Create a sample PDF with a chart for testing
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create sample data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot 1: Line chart
        ax1.plot(x, y1, label='sin(x)', linewidth=2)
        ax1.plot(x, y2, label='cos(x)', linewidth=2)
        ax1.set_title('Trigonometric Functions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Bar chart
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 56, 78, 32]
        ax2.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
        ax2.set_title('Sample Bar Chart')
        ax2.set_xlabel('Categories')
        ax2.set_ylabel('Values')
        
        # Add text description
        fig.text(0.1, 0.02, 'This is a sample PDF with charts and diagrams for testing visual content extraction.', 
                fontsize=10, ha='left')
        
        # Save to PDF
        pdf_path = Path("sample_chart_document.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        
        plt.close(fig)
        print(f"Created sample PDF with charts: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("Matplotlib not available, cannot create sample PDF")
        return None

if __name__ == "__main__":
    # Test the visual content processor
    print("Testing Visual Content Processor...")
    
    if not VISUAL_DEPS_AVAILABLE:
        print("Visual processing dependencies not available")
        print("Please install: pip install easyocr pytesseract pdf2image")
        exit(1)
    
    # Create sample PDF if matplotlib is available
    sample_pdf = create_sample_pdf_with_chart()
    
    # Test with existing PDF
    test_files = [
        "sample_chart_document.pdf",
        "IPCC_AR6_WGII_Chapter03_2pages.pdf",
        "climate_change_parkland.pdf"
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(test_file):
            test_file = file
            break
    
    if test_file:
        print(f"Testing with: {test_file}")
        documents = process_pdf_with_visual_understanding(test_file)
        
        print(f"Processed {len(documents)} documents")
        for i, doc in enumerate(documents[:2]):
            print(f"\nDocument {i+1}:")
            print(f"  Length: {len(doc.text)} characters")
            print(f"  Preview: {doc.text[:200]}...")
            if "visual_content" in doc.metadata:
                vc = doc.metadata["visual_content"]
                print(f"  Charts detected: {vc.get('charts_detected', False)}")
                print(f"  Diagrams detected: {vc.get('diagrams_detected', False)}")
                print(f"  Tables detected: {vc.get('tables_detected', False)}")
    else:
        print("No test PDF files found") 