#!/usr/bin/env python3
"""
RAG MCP Server

A FastMCP server that provides RAG (Retrieval-Augmented Generation) capabilities
for clinical trials data. This server wraps the RAG functionality so it can be
used as an external tool by agents like ReAct_LG.py.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import logging

from fastmcp import FastMCP
from dotenv import load_dotenv

# Add parent directory to sys.path to import RAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from RAG.rag import (
        get_llm_model, 
        get_embedding_model, 
        setup_retriever, 
        fetch_trials_from_mongo, 
        transform_trials_to_documents, 
        create_vectorstore
    )
except ImportError as e:
    print(f"Error importing RAG modules: {e}")
    print("Make sure you're running from the correct directory and RAG modules are available")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-mcp-server")

class RAGSystem:
    def __init__(self, embedding_provider="huggingface"):
        self.embedding_provider = embedding_provider
        self.vectorstore = None
        self.retriever = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the RAG system with clinical trials data"""
        try:
            logger.info("Initializing RAG system...")
            
            # Fetch trials from MongoDB
            logger.info("Fetching trials from MongoDB...")
            trials = fetch_trials_from_mongo()
            if not trials:
                logger.error("No trials found in MongoDB")
                return False
            
            logger.info(f"Found {len(trials)} trials in database")
            
            # Transform to documents
            logger.info("Transforming trials to documents...")
            documents = transform_trials_to_documents(trials, embedding_provider=self.embedding_provider)
            
            # Create vectorstore
            logger.info("Creating vectorstore...")
            self.vectorstore = create_vectorstore(documents, embedding_provider=self.embedding_provider)
            if not self.vectorstore:
                logger.error("Failed to create vectorstore")
                return False
            
            # Set up retriever
            logger.info("Setting up retriever...")
            self.retriever = setup_retriever(self.vectorstore, embedding_provider=self.embedding_provider)
            
            self.initialized = True
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant clinical trials using the RAG system"""
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize RAG system")
        
        try:
            # Use the retriever to get relevant documents
            documents = self.retriever.get_relevant_documents(query)
            
            # Convert to serializable format
            results = []
            for i, doc in enumerate(documents[:top_k]):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'relevance_score', None)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise e
    
    def generate_answer(self, query: str, context_docs: List[Dict], llm_provider="mistral", model_name=None):
        """Generate an answer using the retrieved context and an LLM"""
        try:
            # Get LLM
            llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=0.2)
            
            # Format context from documents
            context = "\n\n".join([
                f"Document {doc['rank']}: {doc['content']}"
                for doc in context_docs
            ])
            
            # Create prompt
            prompt = f"""Based on the following clinical trials information, answer the user's question.

Context from Clinical Trials Database:
{context}

Question: {query}

Please provide a comprehensive answer based on the clinical trials data above. If the information is not sufficient to answer the question, indicate what additional information might be needed.

Answer:"""
            
            # Generate response
            response = llm.invoke(prompt)
            
            # Extract content based on response type
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise e

# Initialize the RAG system
rag_system = RAGSystem()

# Create the FastMCP server
mcp = FastMCP("Clinical Trials RAG Server")

@mcp.tool()
def search_clinical_trials(query: str, top_k: int = 5) -> str:
    """
    Search for relevant clinical trials using semantic similarity.
    
    Args:
        query: The search query about clinical trials
        top_k: Number of most relevant results to return (default: 5, max: 20)
    
    Returns:
        JSON string containing relevant clinical trial documents
    """
    try:
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
            
        results = rag_system.search(query, top_k)
        
        response = {
            "success": True,
            "query": query,
            "num_results": len(results),
            "results": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_clinical_trials: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })

@mcp.tool()
def get_trial_answer(query: str, top_k: int = 5, llm_provider: str = "mistral", model_name: Optional[str] = None) -> str:
    """
    Get an AI-generated answer about clinical trials using RAG.
    
    Args:
        query: The question about clinical trials
        top_k: Number of documents to retrieve for context (default: 5, max: 10)
        llm_provider: LLM provider to use for answer generation
        model_name: Specific model name (optional)
    
    Returns:
        JSON string containing the generated answer and source documents
    """
    try:
        if top_k > 10:
            top_k = 10
        elif top_k < 1:
            top_k = 1
            
        # First, search for relevant documents
        context_docs = rag_system.search(query, top_k)
        
        # Generate answer using the context
        answer = rag_system.generate_answer(query, context_docs, llm_provider, model_name)
        
        response = {
            "success": True,
            "query": query,
            "answer": answer,
            "source_documents": context_docs,
            "llm_provider": llm_provider,
            "model_name": model_name
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_trial_answer: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })

@mcp.tool()
def get_trial_by_nct_id(nct_id: str) -> str:
    """
    Get detailed information about a specific clinical trial by NCT ID.
    
    Args:
        nct_id: The NCT ID of the clinical trial (e.g., NCT12345678)
    
    Returns:
        JSON string containing detailed trial information
    """
    try:
        # Search for the specific NCT ID
        query = f"NCT ID {nct_id}"
        results = rag_system.search(query, top_k=10)
        
        # Filter results to find exact matches
        exact_matches = []
        for result in results:
            if nct_id.upper() in result.get("content", "").upper() or \
               nct_id.upper() in str(result.get("metadata", {})).upper():
                exact_matches.append(result)
        
        response = {
            "success": True,
            "nct_id": nct_id,
            "exact_matches": len(exact_matches),
            "trial_info": exact_matches[:3] if exact_matches else [],
            "related_results": results[:5] if not exact_matches else []
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_trial_by_nct_id: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "nct_id": nct_id
        })

@mcp.tool()
def search_trials_by_condition(condition: str, top_k: int = 10) -> str:
    """
    Search for clinical trials related to a specific medical condition.
    
    Args:
        condition: The medical condition or disease name
        top_k: Number of results to return (default: 10, max: 20)
    
    Returns:
        JSON string containing trials related to the condition
    """
    try:
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
            
        # Enhance query for better condition-based search
        enhanced_query = f"clinical trial {condition} disease condition treatment"
        results = rag_system.search(enhanced_query, top_k)
        
        response = {
            "success": True,
            "condition": condition,
            "search_query": enhanced_query,
            "num_results": len(results),
            "trials": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_trials_by_condition: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "condition": condition
        })

@mcp.tool()
def get_rag_system_status() -> str:
    """
    Get the current status of the RAG system.
    
    Returns:
        JSON string containing system status information
    """
    try:
        status_info = {
            "success": True,
            "initialized": rag_system.initialized,
            "embedding_provider": rag_system.embedding_provider,
            "vectorstore_available": rag_system.vectorstore is not None,
            "retriever_available": rag_system.retriever is not None
        }
        
        # If initialized, get some stats
        if rag_system.initialized and rag_system.vectorstore:
            try:
                # Try to get vectorstore info if available
                if hasattr(rag_system.vectorstore, '_collection'):
                    status_info["vectorstore_type"] = type(rag_system.vectorstore).__name__
                status_info["system_ready"] = True
            except:
                status_info["system_ready"] = False
        else:
            status_info["system_ready"] = False
        
        return json.dumps(status_info, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_rag_system_status: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

def main():
    """Main function to run the RAG MCP server"""
    logger.info("Starting Clinical Trials RAG MCP Server...")
    
    # Pre-initialize the RAG system
    logger.info("Pre-initializing RAG system...")
    if rag_system.initialize():
        logger.info("RAG system ready!")
    else:
        logger.warning("RAG system initialization failed. Will retry on first request.")
    
    # Run the MCP server
    mcp.run()

if __name__ == "__main__":
    main() 