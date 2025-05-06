# add embedding for documents in mongo db

from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import MongoDBLoader
from langchain_core.embeddings import Embeddings
import os

# def embed_documents_hf(embedding_model):
#     # embed documents with embedding_models in huggingface
#     # load documents from mongo db
#     loader = MongoDBLoader(
#         database="clinical_trials",
#         collection="trialgpt_trials",
#     )   
#     documents = loader.load()
#     # embed documents with embedding_models in huggingface and save to mongo db
#     embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#     documents = embeddings.embed_documents(documents)
#     return documents


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
            # from sentence_transformers import SentenceTransformer
            # biobert = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
            # return biobert
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        # elif provider == "topo":
        #     pass
        elif provider == "trial2vec":
            pass
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAIEmbeddings()
        
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

if __name__ == "__main__":
    # embed_documents_hf("sentence-transformers/all-MiniLM-L6-v2")
    # embed_documents_hf("sentence-transformers/all-mpnet-base-v2")
    # etc
    pass

