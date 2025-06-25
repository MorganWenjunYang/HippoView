# add embedding for documents in mongo db

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import MongoDBLoader
from langchain_core.embeddings import Embeddings
from embedding.trial2vec_adapter import Trial2VecEmbeddings
import os
import pandas as pd
import pickle   


def get_embedding_model(provider: str = "huggingface") -> Embeddings:
    """Get embedding model based on provider.
    
    Args:
        provider: The embedding provider (openai, huggingface, cohere, mistral, etc.)
        
    Returns:
        A LangChain embeddings model
    """
    # Try to use specified provider
    try:
        if provider == "biobert":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        # elif provider == "topo":
        #     pass
        elif provider == "trial2vec":
            return Trial2VecEmbeddings()
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAIEmbeddings()
        elif provider == "BGE-M3":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")        
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            # These models run locally without API access
            model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                return HuggingFaceEmbeddings(model_name=model_name)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                print("Trying alternative local model...")
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        elif provider == "cohere":
            from langchain_cohere import CohereEmbeddings
            if not os.getenv("COHERE_API_KEY"):
                raise ValueError("COHERE_API_KEY environment variable not set")
            return CohereEmbeddings(model="embed-english-v3.0")
        
        elif provider == "mistral":
            return get_embedding_model("huggingface")
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
    except (ImportError, ValueError) as e:
        print(f"Warning: Could not use {provider} embeddings: {str(e)}")
        print("Falling back to HuggingFace sentence-transformers (local) embeddings")
        
        # Fallback to HuggingFace (requires no API key)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            print("Error: Could not load HuggingFace embeddings. Please install with:")
            print("pip install langchain-huggingface sentence-transformers")
            raise

def save_embedding(embedding, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embedding, f)

def load_embedding(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    pass

