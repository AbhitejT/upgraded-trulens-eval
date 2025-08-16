import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# LlamaIndex imports for QA generation
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
import openai

logger = logging.getLogger("generate_qa")

def generate_qa_pairs_from_documents(
    document_folder: str,
    output_file: str = "test_data.jsonl",
    num_questions_total: int = 200,
    num_questions_per_source: int = 5,
    openai_api_key: str = None
):
    # Generate QA pairs from documents using LlamaIndex
    
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
    
    # Check if document folder exists and has files
    doc_path = Path(document_folder)
    if not doc_path.exists():
        raise ValueError(f"Document folder '{document_folder}' does not exist")
    
    if not doc_path.is_dir():
        raise ValueError(f"'{document_folder}' is not a directory")
    
    # Check for files in directory
    files = list(doc_path.glob("*"))
    if not files:
        raise ValueError(f"No files found in '{document_folder}'")
    
    # Initialize LlamaIndex components
    llm = OpenAI(model="gpt-4", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    # Load documents
    print(f"Loading documents from {document_folder}...")
    try:
        documents = SimpleDirectoryReader(input_dir=document_folder).load_data()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        raise ValueError(f"Failed to load documents from {document_folder}: {e}")
    
    if not documents:
        raise ValueError("No documents were loaded successfully")
    
    # Convert documents to nodes using a node parser
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
    nodes = []
    for doc in documents:
        nodes.extend(node_parser.get_nodes_from_documents([doc]))
    print(f"Parsed {len(nodes)} nodes from documents")
    
    if not nodes:
        raise ValueError("No nodes were created from documents")
    
    # Generate QA pairs
    print("Generating QA pairs...")
    try:
        qa_pairs = generate_question_context_pairs(
            nodes=nodes,
            llm=llm,
            num_questions_per_chunk=num_questions_per_source
        )
    except Exception as e:
        raise ValueError(f"Failed to generate QA pairs: {e}")
    
    # Save to file
    print(f"Saving QA pairs to {output_file}...")
    with open(output_file, "w") as f:
        for query_id, query in qa_pairs.queries.items():
            relevant_docs = qa_pairs.relevant_docs.get(query_id, [])
            context = ""
            if relevant_docs:
                context = qa_pairs.corpus.get(relevant_docs[0], "")
            
            formatted_pair = {
                "question": query,
                "truth": f"Answer based on: {context[:200]}...",
                "context": context
            }
            f.write(json.dumps(formatted_pair) + "\n")
    
    print(f"QA pairs saved to {output_file}")
    return qa_pairs

def generate_test_qa_data(
    openai_config: dict,
    num_questions_total: int,
    num_questions_per_source: int,
    output_file: Path,
    document_folder: str = "./your_documents"
):
    # Simplified version of Azure's generate_test_qa_data for local documents
    
    logger.info(
        "Generating %d questions total, %d per source, from documents in %s",
        num_questions_total,
        num_questions_per_source,
        document_folder
    )
    
    qa_pairs = generate_qa_pairs_from_documents(
        document_folder=document_folder,
        output_file=str(output_file),
        num_questions_total=num_questions_total,
        num_questions_per_source=num_questions_per_source,
        openai_api_key=openai_config.get("api_key")
    )
    
    logger.info("Generated QA pairs successfully")
    return qa_pairs

def generate_based_on_questions(openai_client, model: str, qa: list, num_questions: int, prompt: str):
    # Generate additional questions based on existing ones
    import random
    
    existing_questions = ""
    if qa:
        qa = random.sample(qa, len(qa))
        existing_questions = "\n".join([item["question"] for item in qa])

    gpt_response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{prompt} Only generate {num_questions} TOTAL. Separate each question by a new line. \n{existing_questions}",
            }
        ],
        n=1,
        max_tokens=num_questions * 50,
        temperature=0.3,
    )

    qa = []
    for message in gpt_response.choices[0].message.content.split("\n")[0:num_questions]:
        qa.append({"question": message, "truth": f"Generated from this prompt: {prompt}"})
    return qa

def generate_dontknows_qa_data(openai_config: dict, num_questions_total: int, input_file: Path, output_file: Path):
    # Generate off-topic questions based on existing QA pairs
    import math
    
    logger.info("Generating off-topic questions based on %s", input_file)
    with open(input_file, encoding="utf-8") as f:
        qa = [json.loads(line) for line in f.readlines()]

    openai_client = openai.OpenAI(api_key=openai_config.get("api_key"))
    dontknows_qa = []
    num_questions_each = math.ceil(num_questions_total / 4)
    
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config.get("model", "gpt-3.5-turbo"),
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions that are very related but are not directly answerable by the same sources. Do not simply ask for other examples of the same thing - your question should be standalone.",
    )
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config.get("model", "gpt-3.5-turbo"),
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions with similar keywords that are about publicly known facts.",
    )
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config.get("model", "gpt-3.5-turbo"),
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions that are not related to these topics at all but have well known answers.",
    )
    remaining = num_questions_total - len(dontknows_qa)
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config.get("model", "gpt-3.5-turbo"),
        qa=None,
        num_questions=remaining,
        prompt=f"Suggest {remaining} questions that are nonsensical, and would result in confusion if you asked it.",
    )

    logger.info("Writing %d off-topic questions to %s", len(dontknows_qa), output_file)
    directory = Path(output_file).parent
    if not directory.exists():
        directory.mkdir(parents=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dontknows_qa:
            f.write(json.dumps(item) + "\n")

# Example usage
if __name__ == "__main__":
    openai_config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4"
    }
    
    generate_test_qa_data(
        openai_config=openai_config,
        num_questions_total=5,
        num_questions_per_source=2,
        output_file=Path("test_data.jsonl"),
        document_folder="./your_documents"
    )
