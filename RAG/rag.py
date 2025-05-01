# rag.py

import os
import sys
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MongoDB connection from utils
from data.utils import connect_to_mongo

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

def transform_trials_to_documents(trials: List[Dict[Any, Any]]) -> List[Document]:
    """Transform MongoDB trials into Langchain Document objects."""
    documents = []
    
    for trial in trials:
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
        
        Conditions: {', '.join(trial.get('condition', [])) if isinstance(trial.get('condition', []), list) else trial.get('condition', 'N/A')}
        Interventions: {', '.join(trial.get('intervention', [])) if isinstance(trial.get('intervention', []), list) else trial.get('intervention', 'N/A')}
        """
        
        # Handle outcomes if available
        if 'outcomes' in trial and trial['outcomes']:
            content += "\nOutcomes:\n"
            for outcome in trial['outcomes']:
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
    
    return documents

def create_vectorstore(documents: List[Document]):
    """Create a vector store from documents using OpenAI embeddings."""
    try:
        # Initialize the embedding model
        embeddings = OpenAIEmbeddings()
        
        # Create the vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

def setup_retriever(vectorstore):
    """Set up the retriever from a vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def create_rag_chain(retriever):
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
    model = ChatOpenAI(temperature=0.2)
    
    # Setup the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    """Main function to set up and test the RAG system."""
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Please set it to use this script.")
        print("You can set it temporarily with: export OPENAI_API_KEY='your-key-here'")
        return
        
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
    print("Creating vector store...")
    vectorstore = create_vectorstore(documents)
    
    if not vectorstore:
        print("Failed to create vector store. Please check your OpenAI API key.")
        return
    
    # Setup retriever
    retriever = setup_retriever(vectorstore)
    
    # Create RAG chain
    rag_chain = create_rag_chain(retriever)
    
    # Interactive query loop
    print("\n=== Clinical Trials RAG System ===")
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

if __name__ == "__main__":
    main()