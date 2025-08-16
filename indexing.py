from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Settings

def build_basic_index(documents, llm, embed_model=OpenAIEmbedding()):
    # Build a basic vector store index from documents
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    basic_index = VectorStoreIndex.from_documents(documents=documents)
    return basic_index

def build_sentence_window_index(documents, llm, embed_model=OpenAIEmbedding()):
    # Build a sentence window index from documents
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original-text"
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    
    sentence_index = VectorStoreIndex.from_documents(documents=documents)
    return sentence_index 