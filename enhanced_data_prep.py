#!/usr/bin/env python3

# Enhanced document loading with PDF processing and image support

import os
from pathlib import Path
from typing import List, Optional, Union
from llama_index.core import SimpleDirectoryReader, Document

# PDF processing imports
try:
    from llama_index.readers.file import PyMuPDFReader, PDFReader
    PDF_READERS_AVAILABLE = True
except ImportError:
    PDF_READERS_AVAILABLE = False

# Vision and image processing imports
try:
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal
    from llama_index.core.schema import ImageDocument
    from PIL import Image
    import base64
    import io
    VISION_DEPS_AVAILABLE = True
except ImportError:
    VISION_DEPS_AVAILABLE = False

def select_pdf_pages(file_path, pages, output_path):
    # Select specific pages from a PDF file
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        new_doc = fitz.open()
        for page in pages:
            new_doc.insert_pdf(doc, from_page=page-1, to_page=page-1)
        new_doc.save(output_path)
        new_doc.close()
        doc.close()
        return True
    except Exception as e:
        print(f"Error selecting PDF pages: {e}")
        return False

def load_documents_basic(input_files=None, input_dir=None):
    # Load documents using basic SimpleDirectoryReader
    if input_dir:
        documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    else:
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
    return documents

def load_documents_with_images(input_files=None, input_dir=None, extract_images=True):
    # Load documents with enhanced PDF processing and optional image extraction
    print(f"Loading documents with image extraction: {extract_images}")
    
    documents = []
    
    # Handle different input types
    if input_dir:
        file_paths = list(Path(input_dir).glob("*"))
    elif input_files:
        file_paths = [Path(f) for f in input_files]
    else:
        raise ValueError("Must provide either input_files or input_dir")
    
    # Process each file
    for file_path in file_paths:
        if file_path.is_file():
            print(f"Processing: {file_path.name}")
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Try enhanced PDF loading first
                    if PDF_READERS_AVAILABLE:
                        try:
                            pdf_docs = load_pdf_with_images(file_path)
                            documents.extend(pdf_docs)
                        except Exception as e:
                            print(f"Enhanced PDF loading failed, trying basic: {e}")
                            pdf_docs = load_pdf_basic(file_path)
                            documents.extend(pdf_docs)
                    else:
                        pdf_docs = load_pdf_basic(file_path)
                        documents.extend(pdf_docs)
                
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    # Load image files
                    img_docs = load_image_file(file_path)
                    documents.extend(img_docs)
                
                else:
                    # Use basic loader for other file types
                    basic_docs = load_documents_basic(input_files=[str(file_path)])
                    documents.extend(basic_docs)
                    
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents

def load_pdf_with_images(pdf_path: Path) -> List[Document]:
    # Load PDF with image extraction using PyMuPDF
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(str(pdf_path))
        documents = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract images
            image_list = page.get_images()
            image_text = ""
            
            if image_list and VISION_DEPS_AVAILABLE:
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Convert to PIL Image for processing
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Use OpenAI Vision to describe the image
                            try:
                                img_description = describe_image_with_vision(img_pil)
                                image_text += f"\n[Image {img_index + 1}]: {img_description}\n"
                            except Exception as e:
                                print(f"Vision processing failed for image {img_index + 1}: {e}")
                                image_text += f"\n[Image {img_index + 1}]: Image detected but could not be processed\n"
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error processing image {img_index + 1}: {e}")
                        continue
            
            # Combine text and image descriptions
            full_text = text
            if image_text:
                full_text += f"\n\n{image_text}"
            
            # Create document
            document = Document(
                text=full_text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "has_images": len(image_list) > 0
                }
            )
            documents.append(document)
        
        doc.close()
        return documents
        
    except ImportError:
        print("PyMuPDF not available, falling back to basic PDF loading")
        return load_pdf_basic(pdf_path)
    except Exception as e:
        print(f"Error loading PDF with images: {e}")
        return load_pdf_basic(pdf_path)

def load_pdf_basic(pdf_path: Path) -> List[Document]:
    # Basic PDF loading using PyPDF
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(str(pdf_path))
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            document = Document(
                text=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num + 1,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(document)
        
        return documents
        
    except ImportError:
        print("PyPDF not available, falling back to SimpleDirectoryReader")
        return load_documents_basic(input_files=[str(pdf_path)])
    except Exception as e:
        print(f"Error in basic PDF loading: {e}")
        return load_documents_basic(input_files=[str(pdf_path)])

def load_image_file(image_path: Path) -> List[Document]:
    # Load image files with vision processing
    if not VISION_DEPS_AVAILABLE:
        print("Vision dependencies not available, skipping image processing")
        return []
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Use OpenAI Vision to describe the image
        try:
            description = describe_image_with_vision(image)
        except Exception as e:
            print(f"Vision processing failed: {e}")
            description = f"Image file: {image_path.name}"
        
        # Create document
        document = Document(
            text=description,
            metadata={
                "source": str(image_path),
                "file_type": "image",
                "dimensions": f"{image.width}x{image.height}",
                "mode": image.mode
            }
        )
        
        return [document]
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

def describe_image_with_vision(image: Image.Image) -> str:
    # Use OpenAI Vision to describe an image
    if not VISION_DEPS_AVAILABLE:
        return f"Image file: {image.size}"
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Use OpenAI Vision API
        client = OpenAIMultiModal(
            model="gpt-4-vision-preview",
            max_new_tokens=300
        )
        
        response = client.complete(
            prompt="Describe this image in detail, focusing on any text, charts, diagrams, or visual data that would be important for document understanding.",
            image_documents=[ImageDocument(image=image)]
        )
        
        return response.text
        
    except Exception as e:
        print(f"OpenAI Vision API error: {e}")
        return f"Image file: {image.size} (vision processing failed)"

def load_documents(input_files=None, input_dir=None, extract_images=True):
    # Main document loading function with automatic fallback handling
    # Attempts to use the most appropriate loading method based on available dependencies
    
    if extract_images and VISION_DEPS_AVAILABLE:
        try:
            return load_documents_with_images(input_files=input_files, input_dir=input_dir, extract_images=True)
        except Exception as e:
            print(f"Enhanced loading failed: {e}")
            print("Falling back to basic loading...")
    
    return load_documents_basic(input_files=input_files, input_dir=input_dir)

# Convenience function for backward compatibility
def load_documents_with_fallback(input_files=None, input_dir=None):
    return load_documents(input_files=input_files, input_dir=input_dir, extract_images=True) 