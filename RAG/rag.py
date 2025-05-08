# rag.py

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from embedding.embd import get_embedding_model
from embedding.trial2vec_adapter import mongodb_data_adaptor
# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MongoDB connection from utils
from data.utils import connect_to_mongo

def get_llm_model(provider: str = "openai", model_name: Optional[str] = None, temperature: float = 0.2) -> BaseChatModel:
    """Get the LLM model based on the provider.
    
    Args:
        provider: The LLM provider (openai, anthropic, huggingface, mistral, gemini)
        model_name: The specific model to use
        temperature: The temperature for generation
        
    Returns:
        A LangChain chat model
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model_name=model_name or "gpt-3.5-turbo", temperature=temperature)
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ChatAnthropic(model_name=model_name or "claude-3-sonnet-20240229", temperature=temperature)
    
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        if not os.getenv("HUGGINGFACE_API_KEY"):
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return HuggingFaceEndpoint(
            endpoint_url=os.getenv("HUGGINGFACE_ENDPOINT_URL", ""),
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
            task="text-generation",
            model_kwargs={"temperature": temperature, "max_length": 512}
        )
    
    elif provider == "mistral":
        from langchain_mistralai.chat_models import ChatMistralAI
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        return ChatMistralAI(model_name=model_name or "mistral-small", temperature=temperature)
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(model_name=model_name or "gemini-pro", temperature=temperature)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def fetch_trials_from_mongo():
    """Fetch clinical trial data from MongoDB."""
    # Use the existing MongoDB connection from utils
    client = connect_to_mongo()
    if not client:
        print("Failed to connect to MongoDB. Check your credentials and connection.")
        return []
    
    try:
        db = client['clinical_trials']
        collection = db['trialgpt_trials']
        
        # Fetch all trials
        trials = list(collection.find())
        return trials
    except Exception as e:
        print(f"Error fetching data from MongoDB: {str(e)}")
        return []
    finally:
        client.close()

def transform_trials_to_documents(trials: List[Dict[Any, Any]], embedding_provider: str = "huggingface") -> List[Document]:
    """Transform MongoDB trials into Langchain Document objects."""
    documents = []

    if embedding_provider == "trial2vec":
        documents = mongodb_data_adaptor(trials)
        return documents
    
    for trial in trials:
        try:
            # Handle conditions - ensure it's a list or convert to string
            conditions = trial.get('condition', [])
            if isinstance(conditions, list):
                conditions_str = ', '.join(conditions)
            elif conditions is None:
                conditions_str = 'N/A'
            else:
                conditions_str = str(conditions)
                
            # Handle interventions - ensure it's a list or convert to string
            interventions = trial.get('intervention', [])
            if isinstance(interventions, list):
                interventions_str = ', '.join(interventions)
            elif interventions is None:
                interventions_str = 'N/A'
            else:
                interventions_str = str(interventions)
            
            # Create a content string from the trial data
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
            
            # Handle outcomes if available
            if 'outcomes' in trial and trial['outcomes'] and isinstance(trial['outcomes'], list):
                content += "\nOutcomes:\n"
                for outcome in trial['outcomes']:
                    if isinstance(outcome, dict):
                        content += f"- Type: {outcome.get('outcome_type', 'N/A')}\n"
                        content += f"  Title: {outcome.get('title', 'N/A')}\n"
                        content += f"  Description: {outcome.get('description', 'N/A')}\n"
                        content += f"  Time Frame: {outcome.get('time_frame', 'N/A')}\n"
            
            # Create Document object
            doc = Document(
                page_content=content,
                metadata={
                    "nct_id": trial.get('nct_id', 'N/A'),
                    "title": trial.get('brief_title', 'N/A'),
                    "status": trial.get('overall_status', 'N/A')
                }
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing trial {trial.get('nct_id', 'unknown')}: {str(e)}")
            # Continue with the next trial instead of failing completely
            continue
    
    return documents

def create_vectorstore(documents: List[Document], embedding_provider: str = "mistral"):
    """Create a vector store from documents using specified embeddings."""
    try:
        # Initialize the embedding model
        embeddings = get_embedding_model(provider=embedding_provider)
        
        # Create the vector store
        batch_size = 100
        vectorstore = None
        
        if embedding_provider == "trial2vec":
            # Convert DataFrame to Document objects using vectorized operations
            df = documents['x']
            print("Available columns:", df.columns.tolist())  # Debug print
            
            # Create content strings for all rows at once
            contents = df.apply(lambda row: f"""
                NCT ID: {row.get('nct_id', 'N/A')}
                Title: {row.get('title', 'N/A')}
                Description: {row.get('description', 'N/A')}
                Intervention: {row.get('intervention_name', 'N/A')}
                Disease: {row.get('disease', 'N/A')}
                Keywords: {row.get('keyword', 'N/A')}
                Outcome Measures: {row.get('outcome_measure', 'N/A')}
                Criteria: {row.get('criteria', 'N/A')}
                Status: {row.get('overall_status', 'N/A')}
                """
                , axis=1)
            
            metadata_list = df.apply(lambda row: {
                "nct_id": row.get('nct_id', 'N/A'),
                "title": row.get('title', 'N/A'),
                "status": row.get('overall_status', 'N/A')
            }, axis=1)
            
            docs = [Document(page_content=content, metadata=metadata) 
                    for content, metadata in zip(contents, metadata_list)]
            
            documents_lc = docs
            texts = [doc.page_content for doc in documents_lc]
            doc_embeddings = embeddings.embed_documents(documents)
            embedding_pairs = zip(texts, doc_embeddings)
            vectorstore = FAISS.from_embeddings(embedding_pairs, embeddings)
            return vectorstore
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                batch_vectorstore = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vectorstore)
        
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

def setup_retriever(vectorstore, embedding_provider: str = "mistral"):
    """Set up the retriever from a vector store."""
    if embedding_provider == "trial2vec":        
        from embedding.trial2vec_adapter import Trial2VecRetriever
        # Get the embedding model
        embeddings = get_embedding_model(provider=embedding_provider)
        return Trial2VecRetriever(vectorstore, embeddings)
    else:
        return vectorstore.as_retriever(search_kwargs={"k": 5})

def create_rag_chain(retriever, llm_provider: str = "mistral", model_name: Optional[str] = None, temperature: float = 0.2):
    """Create a RAG chain with the retriever."""
    # Define the prompt template
    template = """
    You are a clinical trials expert assistant. Use the following clinical trial information to answer the user's question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Setup the language model
    model = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Setup the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clinical Trials RAG System")
    parser.add_argument("--llm", default="mistral", 
                      choices=["openai", "anthropic", "huggingface", "mistral", "gemini"],
                      help="LLM provider to use")
    parser.add_argument("--embedding", default="huggingface", 
                      choices=["openai", "huggingface", "cohere", "mistral"],
                      help="Embedding provider to use")
    parser.add_argument("--model", default=None, 
                      help="Specific model name (provider-dependent)")
    parser.add_argument("--temperature", type=float, default=0.2, 
                      help="Model temperature (0-1)")
    parser.add_argument("--hf-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                      help="HuggingFace embedding model to use")
    return parser.parse_args()

def main():
    """Main function to set up and test the RAG system."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Fetch trials from MongoDB
        print("Fetching trials from MongoDB...")
        trials = fetch_trials_from_mongo()
        
        if not trials:
            print("No trials found in MongoDB. Please check your database connection.")
            return
        
        print(f"Found {len(trials)} trials in MongoDB.")
        
        # Transform trials to documents
        print("Transforming trials to documents...")
        documents = transform_trials_to_documents(trials)
        
        # Create vector store
        print(f"Creating vector store using {args.embedding} embeddings...")
        vectorstore = create_vectorstore(documents, embedding_provider=args.embedding)
        
        if not vectorstore:
            print(f"Failed to create vector store. Trying with fallback embedding model...")
            vectorstore = create_vectorstore(documents, embedding_provider="huggingface")
            
        if not vectorstore:
            print("Failed to create vector store even with fallback. Exiting.")
            return
        
        # Setup retriever
        retriever = setup_retriever(vectorstore, embedding_provider=args.embedding)
        
        # Create RAG chain
        print(f"Setting up RAG chain with {args.llm} model...")
        try:
            rag_chain = create_rag_chain(
                retriever, 
                llm_provider=args.llm,
                model_name=args.model,
                temperature=args.temperature
            )
        except Exception as e:
            print(f"Failed to create RAG chain with {args.llm}: {str(e)}")
            print("Trying with fallback to OpenAI model...")
            
            if os.getenv("OPENAI_API_KEY"):
                rag_chain = create_rag_chain(
                    retriever, 
                    llm_provider="openai",
                    model_name=None,
                    temperature=args.temperature
                )
            else:
                print("Error: Could not use fallback LLM. Please set an API key.")
                return
        
        # Interactive query loop
        print("\n=== Clinical Trials RAG System ===")
        print(f"Using LLM: {args.llm}{f' ({args.model})' if args.model else ''}")
        print(f"Using Embeddings: {args.embedding}")
        if args.embedding == "huggingface":
            print(f"HuggingFace Embedding Model: {os.getenv('HF_EMBEDDING_MODEL')}")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your question about clinical trials: ")
            
            if query.lower() == 'exit':
                break
                
            try:
                # Get response from RAG chain
                response = rag_chain.invoke(query)
                print("\nAnswer:", response)
            except Exception as e:
                print(f"Error processing query: {str(e)}")
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()