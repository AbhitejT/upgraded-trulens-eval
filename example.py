#!/usr/bin/env python3

# Simple example demonstrating the RAG evaluation system

import os
from dotenv import load_dotenv

load_dotenv()

def run_example():
    print("TruLens RAG Evaluation System - Example")
    print("=" * 40)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the .env file")
        return
    
    try:
        # Import core modules
        from enhanced_data_prep import load_documents
        from indexing import build_basic_index
        from evaluation import tru, get_trulens_recorder, get_qa_set_lazy
        from llama_index.llms.openai import OpenAI
        
        print("✅ All modules imported successfully")
        
        # Load sample document
        print("\n📄 Loading sample document...")
        documents = load_documents(input_files=["test_documents/sample.txt"])
        print(f"Loaded {len(documents)} documents")
        
        # Initialize models
        print("\n🤖 Initializing models...")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        print("Models initialized")
        
        # Build index
        print("\n🔍 Building search index...")
        index = build_basic_index(documents=documents, llm=llm)
        query_engine = index.as_query_engine()
        print("Index built successfully")
        
        # Set up evaluation
        print("\n📊 Setting up evaluation...")
        recorder = get_trulens_recorder(query_engine, app_id="Example_RAG_System")
        print("Evaluation setup complete")
        
        # Run a simple test
        print("\n🧪 Running evaluation test...")
        test_question = "What are the key impacts of climate change on oceans?"
        
        with recorder as recording:
            response = query_engine.query(test_question)
            print(f"Question: {test_question}")
            print(f"Response: {str(response)[:200]}...")
        
        print("✅ Evaluation complete!")
        print("\n📈 To view detailed results, run: python dashboard.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set your OpenAI API key in .env")
        print("2. Installed dependencies: pip install -r requirements.txt")
        print("3. Placed documents in test_documents/ folder")

if __name__ == "__main__":
    run_example() 