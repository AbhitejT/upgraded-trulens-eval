#!/usr/bin/env python3

# Main RAG Evaluation Workflow
# Orchestrates the complete RAG evaluation pipeline

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Starting RAG Evaluation Workflow")
    print("=" * 50)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the .env file")
        return
    
    try:
        from enhanced_data_prep import load_documents
        from indexing import build_basic_index
        from evaluation import tru, get_trulens_recorder, get_qa_set_lazy
        from llama_index.llms.openai import OpenAI
        
        print("All modules imported successfully")
        
        # Load documents
        print("\nStep 1: Loading documents...")
        input_files = ["test_documents/sample.txt"]  # Use existing text file
        documents = load_documents(input_files=input_files)
        print(f"Loaded {len(documents)} documents")
        
        # Initialize models
        print("\nStep 2: Initializing models...")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        print("Models initialized")
        
        # Build index
        print("\nStep 3: Building index...")
        basic_query_index = build_basic_index(documents=documents, llm=llm)
        basic_query_engine = basic_query_index.as_query_engine()
        print("Index built successfully")
        
        # Set up evaluation
        print("\nStep 4: Setting up evaluation...")
        basic_recorder = get_trulens_recorder(basic_query_engine, app_id="Basic Query Engine")
        print("Evaluation setup complete")
        
        # Run evaluation
        print("\nStep 5: Running evaluation...")
        with basic_recorder as recording:
            qa_set = get_qa_set_lazy()
            for q in qa_set:
                basic_query_engine.query(q['query'])
        print("Evaluation complete")
        
        # Launch dashboard
        print("\nStep 6: Launching dashboard...")
        print("Press Ctrl+C to stop the dashboard when you're done")
        tru.run_dashboard()
        
    except KeyboardInterrupt:
        print("\nWorkflow stopped by user")
    except Exception as e:
        print(f"Error in workflow: {e}")

if __name__ == "__main__":
    main() 