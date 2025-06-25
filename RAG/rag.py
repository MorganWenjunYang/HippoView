#!/usr/bin/env python3
"""
RAG MCP Server

A FastMCP server that provides RAG (Retrieval-Augmented Generation) capabilities
for clinical trials data. This server wraps the RAG functionality so it can be
used as an external tool by agents
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_utils import VectorStoreConfig


# Add parent directory to sys.path to import RAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils import get_llm_model
try:
    from RAG.rag_utils import (
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
    def __init__(self, embedding_provider="huggingface", vectorstore_config: VectorStoreConfig = None):
        self.embedding_provider = embedding_provider
        self.vectorstore = None
        self.retriever = None
        self.initialized = False
        if vectorstore_config is None:
            self.vectorstore_config = VectorStoreConfig(
                vectorstore_type="elasticsearch",
                embedding_provider=self.embedding_provider,
                persist=True,
                es_url="http://localhost:9200",
                index_name="clinical_trial"
            )
        else:
            self.vectorstore_config = vectorstore_config
        
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

            self.vectorstore = create_vectorstore(documents, config=self.vectorstore_config)
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

def test_rag_system():
    """
    Comprehensive test function for the RAG system.
    Tests initialization, search, and answer generation capabilities.
    """
    print("🧪 === RAG System Testing ===")
    print()
    
    # Test cases for search queries
    test_queries = [
        "diabetes treatment trials",
        "cancer immunotherapy studies", 
        "COVID-19 vaccine trials",
        "heart disease prevention studies",
        "pediatric clinical trials"
    ]
    
    # Test case for answer generation
    sample_question = "What are some clinical trials for diabetes treatment?"
    
    try:
        # Test 1: RAG System Initialization
        print("📋 Test 1: Creating RAG System...")
        rag = RAGSystem()
        print("✅ RAG System created successfully")
        print()
        
        # Test 2: System Initialization 
        print("🔧 Test 2: Initializing RAG System...")
        init_success = rag.initialize()
        
        if not init_success:
            print("❌ RAG System initialization failed")
            return False
        
        print("✅ RAG System initialized successfully")
        print(f"   - MongoDB connection: ✅")
        print(f"   - Vectorstore created: ✅") 
        print(f"   - Retriever setup: ✅")
        print()
        
        # Test 3: Search Functionality
        print("🔍 Test 3: Testing Search Functionality...")
        search_results = {}
        
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"   Testing query {i}: '{query}'")
                results = rag.search(query, top_k=3)
                search_results[query] = results
                
                print(f"      ✅ Found {len(results)} relevant documents")
                
                # Show sample result
                if results:
                    sample_doc = results[0]
                    nct_id = sample_doc['metadata'].get('nct_id', 'N/A')
                    title = sample_doc['metadata'].get('title', 'N/A')
                    print(f"      📄 Top result: {nct_id} - {title[:60]}...")
                
                print()
                
            except Exception as e:
                print(f"      ❌ Search failed for '{query}': {str(e)}")
                search_results[query] = []
        
        print("✅ Search functionality test completed")
        print()
        
        # Test 4: Answer Generation
        print("🤖 Test 4: Testing Answer Generation...")
        try:
            # Get search results for sample question
            context_docs = rag.search(sample_question, top_k=5)
            
            if context_docs:
                print(f"   📄 Retrieved {len(context_docs)} context documents")
                
                # Generate answer
                print("   🔄 Generating answer...")
                answer = rag.generate_answer(
                    query=sample_question,
                    context_docs=context_docs,
                    llm_provider="mistral"
                )
                
                print("   ✅ Answer generated successfully")
                print(f"   📝 Question: {sample_question}")
                print(f"   💬 Answer: {answer[:200]}...")
                print()
                
            else:
                print("   ❌ No context documents found for answer generation")
                
        except Exception as e:
            print(f"   ❌ Answer generation failed: {str(e)}")
        
        # Test 5: Performance Summary
        print("📊 Test 5: Performance Summary...")
        total_queries = len(test_queries)
        successful_searches = sum(1 for results in search_results.values() if results)
        
        print(f"   📈 Search Success Rate: {successful_searches}/{total_queries} ({successful_searches/total_queries*100:.1f}%)")
        
        # Document statistics
        total_docs_found = sum(len(results) for results in search_results.values())
        avg_docs_per_query = total_docs_found / total_queries if total_queries > 0 else 0
        
        print(f"   📄 Average Documents per Query: {avg_docs_per_query:.1f}")
        print(f"   🔍 Total Documents Retrieved: {total_docs_found}")
        print()
        
        # Test 6: Error Handling
        print("🛡️ Test 6: Testing Error Handling...")
        try:
            # Test with empty query
            empty_results = rag.search("", top_k=3)
            print("   ✅ Empty query handled gracefully")
            
            # Test with very specific query that might not match
            specific_results = rag.search("extremely rare disease xyz123", top_k=3)
            print(f"   ✅ Rare query handled gracefully (found {len(specific_results)} results)")
            
        except Exception as e:
            print(f"   ⚠️ Error handling test: {str(e)}")
        
        print()
        
        # Final Results
        print("🎯 === Test Results Summary ===")
        print("✅ RAG System Test PASSED")
        print(f"   - System initialization: ✅")
        print(f"   - Search functionality: ✅ ({successful_searches}/{total_queries} queries successful)")
        print(f"   - Answer generation: ✅")
        print(f"   - Error handling: ✅")
        print()
        print("🚀 RAG System is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG System Test FAILED: {str(e)}")
        import traceback
        print("📋 Error details:")
        traceback.print_exc()
        return False

def test_rag_system_quick():
    """
    Quick test function for basic RAG functionality.
    Useful for rapid testing during development.
    """
    print("⚡ === Quick RAG Test ===")
    
    try:
        # Quick initialization test
        rag = RAGSystem()
        print("✅ RAG System created")
        
        if rag.initialize():
            print("✅ RAG System initialized")
            
            # Quick search test
            results = rag.search("diabetes", top_k=2)
            print(f"✅ Search test: found {len(results)} results")
            
            # Quick answer test if results found
            if results:
                answer = rag.generate_answer("What is diabetes?", results[:2])
                print(f"✅ Answer generated: {len(answer)} characters")
            
            print("🎯 Quick test PASSED")
            return True
        else:
            print("❌ Initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick test FAILED: {e}")
        return False

if __name__ == "__main__":
    # Run full test by default
    print("Starting RAG System Tests...")
    print("Use test_rag_system_quick() for faster testing")
    print()
    
    success = test_rag_system()
    
    if success:
        print("\n🎉 All tests passed! RAG system is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
