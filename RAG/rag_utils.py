#!/usr/bin/env python3
"""
RAG Utilities - Modular RAG components for Clinical Trials

This module provides clean separation between:
1. Embedding creation 
2. Vector store creation (FAISS, ElasticSearch, etc.)
3. Retriever setup

Easy switching between different vector store backends.
"""

import os
import sys
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from tqdm import tqdm

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from embedding.embd import get_embedding_model
from embedding.trial2vec_adapter import mongodb_data_adaptor, Trial2VecRetriever
from data.utils import connect_to_mongo

try:
    from llm_utils import get_llm_model
except ImportError:
    print("Warning: llm_utils not found. Some functionality may be limited.")
    get_llm_model = None

# ============================================================================
# 1. DATA FETCHING AND TRANSFORMATION
# ============================================================================

def fetch_trials_from_mongo() -> List[Dict[Any, Any]]:
    """Fetch clinical trial data from MongoDB."""
    client = connect_to_mongo()
    if not client:
        print("Failed to connect to MongoDB. Check your credentials and connection.")
        return []
    
    try:
        db = client['clinical_trials']
        collection = db['trialgpt_trials']
        trials = list(collection.find())
        print(f"Fetched {len(trials)} trials from MongoDB")
        return trials
    except Exception as e:
        print(f"Error fetching data from MongoDB: {str(e)}")
        return []
    finally:
        client.close()

def transform_trials_to_documents(trials: List[Dict[Any, Any]], embedding_provider: str = "huggingface") -> Union[List[Document], Dict]:
    """Transform MongoDB trials into LangChain Document objects.
    
    Args:
        trials: List of trial dictionaries from MongoDB
        embedding_provider: Provider for embeddings
        
    Returns:
        List of Document objects or dict for trial2vec
    """
    if embedding_provider == "trial2vec":
        return mongodb_data_adaptor(trials)
    
    documents = []
    for trial in tqdm(trials, desc="Transforming trials to documents", unit="trial"):
        try:
            # Handle conditions
            conditions = trial.get('condition', [])
            if isinstance(conditions, list):
                conditions_str = ', '.join(conditions)
            elif conditions is None:
                conditions_str = 'N/A'
            else:
                conditions_str = str(conditions)
                
            # Handle interventions
            interventions = trial.get('intervention', [])
            if isinstance(interventions, list):
                interventions_str = ', '.join(interventions)
            elif interventions is None:
                interventions_str = 'N/A'
            else:
                interventions_str = str(interventions)
            
            # Create content string
            content = f"""
            NCT ID: {trial.get('nct_id', 'N/A')}
            Title: {trial.get('brief_title', 'N/A')}
            Status: {trial.get('overall_status', 'N/A')}
            Phase: {trial.get('phase', 'N/A')}
            Study Type: {trial.get('study_type', 'N/A')}
            
            Brief Summary: {trial.get('brief_summary', 'N/A')}
            
            Start Date: {trial.get('start_date', 'N/A')}
            Completion Date: {trial.get('completion_date', 'N/A')}
            
            Eligibility Criteria: {trial.get('criteria', 'N/A')}
            Gender: {trial.get('gender', 'N/A')}
            Min Age: {trial.get('minimum_age', 'N/A')}
            Max Age: {trial.get('maximum_age', 'N/A')}
            
            Conditions: {conditions_str}
            Interventions: {interventions_str}
            """
            
            # Handle outcomes
            if 'outcomes' in trial and trial['outcomes'] and isinstance(trial['outcomes'], list):
                content += "\nOutcomes:\n"
                for outcome in trial['outcomes']:
                    if isinstance(outcome, dict):
                        content += f"- Type: {outcome.get('outcome_type', 'N/A')}\n"
                        content += f"  Title: {outcome.get('title', 'N/A')}\n"
                        content += f"  Description: {outcome.get('description', 'N/A')}\n"
                        content += f"  Time Frame: {outcome.get('time_frame', 'N/A')}\n"
            
            # Create Document
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "nct_id": trial.get('nct_id', 'N/A'),
                    "title": trial.get('brief_title', 'N/A'),
                    "status": trial.get('overall_status', 'N/A')
                }
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing trial {trial.get('nct_id', 'unknown')}: {str(e)}")
            continue
    
    print(f"Transformed {len(documents)} trials to documents")
    return documents

# ============================================================================
# 2. EMBEDDING CREATION (SEPARATED)
# ============================================================================

def create_embeddings(embedding_provider: str = "huggingface", **kwargs) -> Embeddings:
    """Create an embedding model instance.
    
    Args:
        embedding_provider: Provider for embeddings
        **kwargs: Additional arguments for embedding model
        
    Returns:
        Embeddings instance
    """
    try:
        embeddings = get_embedding_model(provider=embedding_provider, **kwargs)
        print(f"Created embedding model: {embedding_provider}")
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise

# ============================================================================
# 3. VECTOR STORE CREATION (FACTORY PATTERN)
# ============================================================================

class VectorStoreConfig:
    """Configuration class for vector stores."""
    
    def __init__(self, 
                 vectorstore_type: str = "faiss",
                 embedding_provider: str = "huggingface",
                 persist: bool = True,
                 **kwargs):
        self.vectorstore_type = vectorstore_type.lower()
        self.embedding_provider = embedding_provider
        self.persist = persist
        self.kwargs = kwargs

class VectorStoreFactory:
    """Factory for creating different types of vector stores."""
    
    @staticmethod
    def create_faiss_vectorstore(documents: Union[List[Document], Dict], 
                                embeddings: Embeddings,
                                embedding_provider: str,
                                persist: bool = True,
                                cache_dir: str = "faiss_cache") -> FAISS:
        """Create FAISS vector store with optional persistence."""
        
        if persist:
            return VectorStoreFactory._create_persistent_faiss(
                documents, embeddings, embedding_provider, cache_dir
            )
        else:
            return VectorStoreFactory._create_memory_faiss(
                documents, embeddings, embedding_provider
            )
    
    @staticmethod
    def _create_persistent_faiss(documents: Union[List[Document], Dict], 
                                embeddings: Embeddings,
                                embedding_provider: str,
                                cache_dir: str) -> FAISS:
        """Create persistent FAISS with caching."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        # Generate cache key
        cache_key = VectorStoreFactory._generate_cache_key(documents, embedding_provider)
        cache_file = cache_path / f"{cache_key}.faiss"
        metadata_file = cache_path / f"{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists() and metadata_file.exists():
            try:
                print(f"Loading FAISS index from cache: {cache_file}")
                vectorstore = FAISS.load_local(
                    str(cache_path), 
                    embeddings,
                    index_name=cache_key,
                    allow_dangerous_deserialization=True
                )
                
                # Check metadata
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                age_hours = (datetime.now() - metadata['created_at']).total_seconds() / 3600
                print(f"Loaded cached FAISS index (age: {age_hours:.1f}h)")
                return vectorstore
                
            except Exception as e:
                print(f"Error loading cache, recreating: {e}")
        
        # Create new vector store
        print("Creating new FAISS vector store...")
        vectorstore = VectorStoreFactory._create_memory_faiss(documents, embeddings, embedding_provider)
        
        # Save to cache
        try:
            vectorstore.save_local(str(cache_path), index_name=cache_key)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now(),
                'embedding_provider': embedding_provider,
                'document_count': len(documents) if isinstance(documents, list) else len(documents.get('x', []))
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"Saved FAISS index to cache: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
        
        return vectorstore
    
    @staticmethod
    def _create_memory_faiss(documents: Union[List[Document], Dict], 
                           embeddings: Embeddings,
                           embedding_provider: str) -> FAISS:
        """Create in-memory FAISS vector store."""
        
        if embedding_provider == "trial2vec":
            # Handle trial2vec special case
            df = documents['x']
            print("Available columns:", df.columns.tolist())
            
            # Create content strings with progress bar
            tqdm.pandas(desc="Processing trial2vec content")
            contents = df.progress_apply(lambda row: f"""
                NCT ID: {row.get('nct_id', 'N/A')}
                Title: {row.get('title', 'N/A')}
                Description: {row.get('description', 'N/A')}
                Intervention: {row.get('intervention_name', 'N/A')}
                Disease: {row.get('disease', 'N/A')}
                Keywords: {row.get('keyword', 'N/A')}
                Outcome Measures: {row.get('outcome_measure', 'N/A')}
                Criteria: {row.get('criteria', 'N/A')}
                Status: {row.get('overall_status', 'N/A')}
                """.strip(), axis=1)
            
            tqdm.pandas(desc="Processing trial2vec metadata")
            metadata_list = df.progress_apply(lambda row: {
                "nct_id": row.get('nct_id', 'N/A'),
                "title": row.get('title', 'N/A'),
                "status": row.get('overall_status', 'N/A')
            }, axis=1)
            
            print("Creating documents from trial2vec data...")
            docs = [Document(page_content=content, metadata=metadata) 
                    for content, metadata in tqdm(zip(contents, metadata_list), 
                                                 desc="Creating documents", 
                                                 total=len(contents))]
            
            # Generate embeddings using trial2vec
            texts = [doc.page_content for doc in docs]
            print("Generating embeddings with trial2vec...")
            doc_embeddings = embeddings.embed_documents(documents)
            embedding_pairs = list(zip(texts, doc_embeddings))
            
            return FAISS.from_embeddings(embedding_pairs, embeddings)
        
        else:
            # Standard document processing
            batch_size = 100
            vectorstore = None
            
            # Create progress bar for batch processing
            total_batches = (len(documents) + batch_size - 1) // batch_size
            batch_range = range(0, len(documents), batch_size)
            
            with tqdm(batch_range, desc="Creating vectorstore", unit="batch", total=total_batches) as pbar:
                for i in pbar:
                    batch = documents[i:i+batch_size]
                    pbar.set_postfix(docs=f"{min(i+batch_size, len(documents))}/{len(documents)}")
                    
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        batch_vectorstore = FAISS.from_documents(batch, embeddings)
                        vectorstore.merge_from(batch_vectorstore)
            
            return vectorstore
    
    @staticmethod
    def create_elasticsearch_vectorstore(documents: Union[List[Document], Dict],
                                       embeddings: Embeddings,
                                       embedding_provider: str,
                                       **kwargs) -> 'ElasticsearchStore':
        """Create ElasticSearch vector store."""
        try:
            from langchain_community.vectorstores import ElasticsearchStore
            
            # Default ElasticSearch configuration
            es_config = {
                'es_url': kwargs.get('es_url', 'http://localhost:9200'),
                'index_name': kwargs.get('index_name', 'clinical_trials'),
                'embedding': embeddings,
                'es_user': kwargs.get('es_user'),
                'es_password': kwargs.get('es_password'),
                'es_api_key': kwargs.get('es_api_key'),
                'es_cloud_id': kwargs.get('es_cloud_id')
            }
            
            # Remove None values
            es_config = {k: v for k, v in es_config.items() if v is not None}

            print(es_config)
            
            print(f"Creating ElasticSearch vector store: {es_config['es_url']}")
            
            if embedding_provider == "trial2vec":
                # Convert trial2vec format to documents
                df = documents['x']
                print("Processing trial2vec data for ElasticSearch...")
                
                tqdm.pandas(desc="Processing ES trial2vec content")
                contents = df.progress_apply(lambda row: f"""
                    NCT ID: {row.get('nct_id', 'N/A')}
                    Title: {row.get('title', 'N/A')}
                    Description: {row.get('description', 'N/A')}
                    Intervention: {row.get('intervention_name', 'N/A')}
                    Disease: {row.get('disease', 'N/A')}
                    Keywords: {row.get('keyword', 'N/A')}
                    Outcome Measures: {row.get('outcome_measure', 'N/A')}
                    Criteria: {row.get('criteria', 'N/A')}
                    Status: {row.get('overall_status', 'N/A')}
                    """.strip(), axis=1)
                
                tqdm.pandas(desc="Processing ES trial2vec metadata")
                metadata_list = df.progress_apply(lambda row: {
                    "nct_id": row.get('nct_id', 'N/A'),
                    "title": row.get('title', 'N/A'),
                    "status": row.get('overall_status', 'N/A')
                }, axis=1)
                
                docs = [Document(page_content=content, metadata=metadata) 
                        for content, metadata in tqdm(zip(contents, metadata_list),
                                                     desc="Creating ES documents",
                                                     total=len(contents))]
                documents = docs
            
            # Create vector store
            vectorstore = ElasticsearchStore.from_documents(
                documents=documents,
                **es_config
            )

            
            print(f"Created ElasticSearch vector store with {len(documents)} documents")
            return vectorstore
            
        except ImportError:
            raise ImportError("langchain_elasticsearch not installed. Install with: pip install langchain-elasticsearch")
        except Exception as e:
            print(f"Error creating ElasticSearch vector store: {e}")
            raise
    
    @staticmethod
    def _generate_cache_key(documents: Union[List[Document], Dict], embedding_provider: str) -> str:
        """Generate cache key for documents."""
        hasher = hashlib.md5()
        hasher.update(embedding_provider.encode())
        
        if isinstance(documents, dict):
            # trial2vec case
            doc_count = len(documents.get('x', []))
            hasher.update(str(doc_count).encode())
        else:
            # Regular documents
            doc_count = len(documents)
            hasher.update(str(doc_count).encode())
            # Sample first few documents for hash
            for doc in documents[:min(5, len(documents))]:
                hasher.update(doc.page_content[:100].encode('utf-8', errors='ignore'))
        
        return f"{embedding_provider}_{hasher.hexdigest()[:12]}"

def create_vectorstore(documents: Union[List[Document], Dict],
                      config: VectorStoreConfig) -> VectorStore:
    """Factory function to create vector stores.
    
    Args:
        documents: Documents to vectorize
        config: Vector store configuration
        
    Returns:
        Vector store instance
    """
    # Create embeddings
    embeddings = create_embeddings(config.embedding_provider)
    
    # Create vector store based on type
    if config.vectorstore_type == "faiss":
        return VectorStoreFactory.create_faiss_vectorstore(
            documents=documents,
            embeddings=embeddings,
            embedding_provider=config.embedding_provider,
            persist=config.persist,
            **config.kwargs
        )
    
    elif config.vectorstore_type == "elasticsearch":
        return VectorStoreFactory.create_elasticsearch_vectorstore(
            documents=documents,
            embeddings=embeddings,
            embedding_provider=config.embedding_provider,
            **config.kwargs
        )
    
    else:
        raise ValueError(f"Unsupported vector store type: {config.vectorstore_type}")

# ============================================================================
# 4. RETRIEVER SETUP (ABSTRACTED)
# ============================================================================

def setup_retriever(vectorstore: VectorStore, 
                   embedding_provider: str = "huggingface",
                   search_kwargs: Optional[Dict] = None) -> BaseRetriever:
    """Set up retriever from vector store.
    
    Args:
        vectorstore: Vector store instance
        embedding_provider: Embedding provider used
        search_kwargs: Search configuration
        
    Returns:
        Retriever instance
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    
    if embedding_provider == "trial2vec":
        # Special handling for trial2vec
        embeddings = create_embeddings(embedding_provider)
        return Trial2VecRetriever(vectorstore, embeddings)
    else:
        # Standard retriever
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

# ============================================================================
# 5. RAG CHAIN CREATION
# ============================================================================

def create_rag_chain(retriever: BaseRetriever, 
                    llm_provider: str = "mistral", 
                    model_name: Optional[str] = None, 
                    temperature: float = 0.2):
    """Create a RAG chain with the retriever.
    
    Args:
        retriever: Document retriever
        llm_provider: LLM provider
        model_name: Specific model name
        temperature: Model temperature
        
    Returns:
        RAG chain
    """
    if get_llm_model is None:
        raise ImportError("llm_utils not available. Cannot create RAG chain.")
    
    # Define prompt template
    template = """
    You are a clinical trials expert assistant. Use the following clinical trial information to answer the user's question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Setup language model
    model = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# ============================================================================
# 6. CONVENIENCE FUNCTIONS
# ============================================================================

def create_clinical_trials_rag(vectorstore_type: str = "faiss",
                              embedding_provider: str = "huggingface",
                              persist: bool = True,
                              llm_provider: str = "mistral",
                              **kwargs):
    """One-stop function to create a complete clinical trials RAG system.
    
    Args:
        vectorstore_type: Type of vector store ("faiss", "elasticsearch")
        embedding_provider: Embedding provider
        persist: Whether to persist vector store
        llm_provider: LLM provider
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (rag_chain, vectorstore, retriever)
    """
    print("=== Creating Clinical Trials RAG System ===")
    
    # Initialize progress tracking
    steps = [
        "Fetching trials from MongoDB",
        "Transforming trials to documents", 
        "Creating vector store",
        "Setting up retriever",
        "Creating RAG chain"
    ]
    
    with tqdm(total=len(steps), desc="RAG System Setup", unit="step") as pbar:
        # Fetch and transform data
        pbar.set_description("Fetching trials from MongoDB")
        trials = fetch_trials_from_mongo()
        if not trials:
            raise ValueError("No trials found in MongoDB")
        pbar.update(1)
        
        pbar.set_description("Transforming trials to documents")
        documents = transform_trials_to_documents(trials, embedding_provider)
        pbar.update(1)
        
        # Create vector store
        pbar.set_description("Creating vector store")
        config = VectorStoreConfig(
            vectorstore_type=vectorstore_type,
            embedding_provider=embedding_provider,
            persist=persist,
            **kwargs
        )
        vectorstore = create_vectorstore(documents, config)
        pbar.update(1)
        
        # Setup retriever
        pbar.set_description("Setting up retriever")
        retriever = setup_retriever(vectorstore, embedding_provider)
        pbar.update(1)
        
        # Create RAG chain
        pbar.set_description("Creating RAG chain")
        rag_chain = create_rag_chain(retriever, llm_provider)
        pbar.update(1)
    
    print("✅ Clinical Trials RAG system ready!")
    return rag_chain, vectorstore, retriever

def get_available_vectorstore_types() -> List[str]:
    """Get list of available vector store types."""
    types = ["faiss"]
    
    try:
        from langchain_community.vectorstores import ElasticsearchStore
        types.append("elasticsearch")
    except ImportError:
        pass
    
    return types

def get_vectorstore_info(vectorstore: VectorStore) -> Dict[str, Any]:
    """Get information about a vector store."""
    info = {
        "type": type(vectorstore).__name__,
        "module": type(vectorstore).__module__
    }
    
    # Add type-specific info
    if isinstance(vectorstore, FAISS):
        info.update({
            "index_type": "FAISS",
            "in_memory": True
        })
    
    try:
        from langchain_community.vectorstores import ElasticsearchStore
        if isinstance(vectorstore, ElasticsearchStore):
            info.update({
                "index_type": "ElasticSearch",
                "distributed": True,
                "persistent": True
            })
    except ImportError:
        pass
    
    return info

# ============================================================================
# 7. TESTING AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=== RAG Utils Test ===")
    
    # Test FAISS
    print("\n1. Testing FAISS vector store...")
    try:
        rag_chain_faiss, vectorstore_faiss, retriever_faiss = create_clinical_trials_rag(
            vectorstore_type="faiss",
            embedding_provider="trial2vec",
            persist=False
        )
        print("✅ FAISS RAG system created successfully")
        print("Vector store info:", get_vectorstore_info(vectorstore_faiss))
        
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
    
    # Test ElasticSearch (if available)
    print("\n2. Testing ElasticSearch vector store...")
    try:
        rag_chain_es, vectorstore_es, retriever_es = create_clinical_trials_rag(
            vectorstore_type="elasticsearch",
            embedding_provider="huggingface",
            es_url="http://localhost:9200",
            index_name="test_clinical_trials"
        )
        print("✅ ElasticSearch RAG system created successfully")
        print("Vector store info:", get_vectorstore_info(vectorstore_es))
        
    except Exception as e:
        print(f"❌ ElasticSearch test failed: {e}")
    
    print(f"\nAvailable vector store types: {get_available_vectorstore_types()}")

    


