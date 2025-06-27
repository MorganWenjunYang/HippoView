#!/usr/bin/env python3
"""
Unified Clinical Trials MCP Server

A comprehensive MCP server that provides both vector-based and graph-based RAG capabilities 
for clinical trials data. All business logic is handled at the MCP layer.

Available Tools:
- Vector-based RAG tools (Elasticsearch/vector search)
- Graph-based RAG tools (Neo4j/graph search)
- System status and health checks
"""

import json
import logging
import os
import sys
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import both RAG systems
from RAG.rag import RAGSystem
from RAG.graphrag import GraphDB, DEFAULT_GRAPHDB_CONFIG
from llm_utils import get_llm_model

# Initialize both systems
from RAG.rag_utils import VectorStoreConfig

VECTORSTORE_CONFIG = VectorStoreConfig(
    vectorstore_type="elasticsearch",
    embedding_provider="huggingface",
    persist=True,
    es_url="http://localhost:9200",
    index_name="copd"
)

vector_rag_system = RAGSystem(embedding_provider="huggingface", vectorstore_config=VECTORSTORE_CONFIG)
graph_db = GraphDB(DEFAULT_GRAPHDB_CONFIG)

# Create the FastMCP server
mcp = FastMCP("Unified Clinical Trials RAG Server")

def initialize_vector_system():
    """Initialize the vector-based RAG system"""
    if not vector_rag_system.initialized:
        return vector_rag_system.initialize()
    return True

def initialize_graph_connection():
    """Initialize the graph database connection"""
    if not graph_db.connected:
        return graph_db.connect()
    return True

def format_trial_content(trial_data: Dict) -> str:
    """Format trial data into readable content"""
    content_parts = []
    
    if trial_data.get("title"):
        content_parts.append(f"Title: {trial_data['title']}")
    
    if trial_data.get("brief_summary"):
        content_parts.append(f"Summary: {trial_data['brief_summary']}")
    
    if trial_data.get("condition"):
        content_parts.append(f"Condition: {trial_data['condition']}")
    
    if trial_data.get("intervention_name"):
        content_parts.append(f"Intervention: {trial_data['intervention_name']}")
    
    if trial_data.get("phase"):
        content_parts.append(f"Phase: {trial_data['phase']}")
    
    if trial_data.get("status"):
        content_parts.append(f"Status: {trial_data['status']}")
    
    return "\n".join(content_parts)

def format_search_results(results: List[Dict]) -> List[Dict[str, Any]]:
    """Format query results into standardized format"""
    formatted_results = []
    for i, result in enumerate(results):
        formatted_result = {
            "rank": i + 1,
            "nct_id": result.get("nct_id", ""),
            "title": result.get("title", ""),
            "brief_summary": result.get("brief_summary", ""),
            "condition": result.get("condition", ""),
            "intervention_name": result.get("intervention_name", ""),
            "phase": result.get("phase", ""),
            "status": result.get("status", ""),
            "content": format_trial_content(result),
            "metadata": {
                "nct_id": result.get("nct_id", ""),
                "title": result.get("title", ""),
                "phase": result.get("phase", ""),
                "status": result.get("status", ""),
                "condition": result.get("condition", ""),
                "intervention_name": result.get("intervention_name", "")
            }
        }
        formatted_results.append(formatted_result)
    
    return formatted_results

def determine_search_type(query: str) -> str:
    """Intelligently determine search type based on query content"""
    query_lower = query.lower()
    
    # Check for condition keywords
    condition_keywords = ["diabetes", "cancer", "heart", "covid", "alzheimer", "depression", 
                         "hypertension", "stroke", "pneumonia", "asthma", "arthritis"]
    if any(condition in query_lower for condition in condition_keywords):
        return "condition"
    
    # Check for intervention keywords
    intervention_keywords = ["drug", "therapy", "treatment", "vaccine", "surgery", 
                           "medication", "pharmaceutical", "dose", "dosage"]
    if any(intervention in query_lower for intervention in intervention_keywords):
        return "intervention"
    
    # Default to text search
    return "text"

# =============================================================================
# VECTOR-BASED RAG TOOLS (Original RAG System)
# =============================================================================

@mcp.tool()
def search_clinical_trials(query: str, top_k: int = 5) -> str:
    """
    Search for relevant clinical trials using vector-based semantic similarity.
    
    Args:
        query: The search query about clinical trials
        top_k: Number of most relevant results to return (default: 5, max: 20)
    
    Returns:
        JSON string containing relevant clinical trial documents
    """
    try:
        if not initialize_vector_system():
            return json.dumps({
                "success": False,
                "error": "Failed to initialize vector RAG system",
                "query": query
            })
        
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
            
        results = vector_rag_system.search(query, top_k)
        
        response = {
            "success": True,
            "query": query,
            "search_method": "vector_similarity",
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
def get_trial_answer(query: str, top_k: int = 5, llm_provider: str = "deepseek", model_name: Optional[str] = None) -> str:
    """
    Get an AI-generated answer about clinical trials using vector-based RAG.
    
    Args:
        query: The question about clinical trials
        top_k: Number of documents to retrieve for context (default: 5, max: 10)
        llm_provider: LLM provider to use for answer generation
        model_name: Specific model name (optional)
    
    Returns:
        JSON string containing the generated answer and source documents
    """
    try:
        if not initialize_vector_system():
            return json.dumps({
                "success": False,
                "error": "Failed to initialize vector RAG system",
                "query": query
            })
        
        if top_k > 10:
            top_k = 10
        elif top_k < 1:
            top_k = 1
            
        # First, search for relevant documents
        context_docs = vector_rag_system.search(query, top_k)
        
        # Generate answer using the context
        answer = vector_rag_system.generate_answer(query, context_docs, llm_provider, model_name)
        
        response = {
            "success": True,
            "query": query,
            "answer": answer,
            "source_documents": context_docs,
            "search_method": "vector_similarity",
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
    Get detailed information about a specific clinical trial by NCT ID using vector search.
    
    Args:
        nct_id: The NCT ID of the clinical trial (e.g., NCT12345678)
    
    Returns:
        JSON string containing detailed trial information
    """
    try:
        if not initialize_vector_system():
            return json.dumps({
                "success": False,
                "error": "Failed to initialize vector RAG system",
                "nct_id": nct_id
            })
        
        # Search for the specific NCT ID
        query = f"NCT ID {nct_id}"
        results = vector_rag_system.search(query, top_k=10)
        
        # Filter results to find exact matches
        exact_matches = []
        for result in results:
            if nct_id.upper() in result.get("content", "").upper() or \
               nct_id.upper() in str(result.get("metadata", {})).upper():
                exact_matches.append(result)
        
        response = {
            "success": True,
            "nct_id": nct_id,
            "search_method": "vector_similarity",
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
def search_trials_by_condition_vector(condition: str, top_k: int = 10) -> str:
    """
    Search for clinical trials related to a specific medical condition using vector similarity.
    
    Args:
        condition: The medical condition or disease name
        top_k: Number of results to return (default: 10, max: 20)
    
    Returns:
        JSON string containing trials related to the condition
    """
    try:
        if not initialize_vector_system():
            return json.dumps({
                "success": False,
                "error": "Failed to initialize vector RAG system",
                "condition": condition
            })
        
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
            
        # Enhance query for better condition-based search
        enhanced_query = f"clinical trial {condition} disease condition treatment"
        results = vector_rag_system.search(enhanced_query, top_k)
        
        response = {
            "success": True,
            "condition": condition,
            "search_method": "vector_similarity",
            "search_query": enhanced_query,
            "num_results": len(results),
            "trials": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_trials_by_condition_vector: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "condition": condition
        })

# =============================================================================
# GRAPH-BASED RAG TOOLS (Neo4j GraphDB System)
# =============================================================================

@mcp.tool()
def search_clinical_trials_graph(query: str, top_k: int = 5, search_type: str = "auto") -> str:
    """
    Search for relevant clinical trials using graph-based search.
    
    Args:
        query: The search query about clinical trials
        top_k: Number of most relevant results to return (default: 5, max: 20)
        search_type: Type of search ("auto", "text", "condition", "intervention")
    
    Returns:
        JSON string containing relevant clinical trial documents
    """
    try:
        # Initialize connection if needed
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database",
                "query": query
            })
        
        # Validate parameters
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
        
        # Determine search type
        if search_type == "auto":
            search_type = determine_search_type(query)
        
        # Execute appropriate search
        if search_type == "condition":
            cypher_query = """
            MATCH (t:ClinicalTrial)
            WHERE toLower(t.condition) CONTAINS toLower($query)
            RETURN t.nct_id as nct_id,
                   t.title as title,
                   t.brief_summary as brief_summary,
                   t.condition as condition,
                   t.intervention_name as intervention_name,
                   t.phase as phase,
                   t.status as status,
                   t.enrollment as enrollment
            ORDER BY t.enrollment DESC
            LIMIT $limit
            """
        elif search_type == "intervention":
            cypher_query = """
            MATCH (t:ClinicalTrial)
            WHERE toLower(t.intervention_name) CONTAINS toLower($query)
               OR toLower(t.intervention_description) CONTAINS toLower($query)
            RETURN t.nct_id as nct_id,
                   t.title as title,
                   t.brief_summary as brief_summary,
                   t.intervention_name as intervention_name,
                   t.intervention_description as intervention_description,
                   t.phase as phase,
                   t.status as status,
                   t.condition as condition
            ORDER BY t.phase DESC
            LIMIT $limit
            """
        else:  # text search
            cypher_query = """
            MATCH (t:ClinicalTrial)
            WHERE toLower(t.title) CONTAINS toLower($query)
               OR toLower(t.brief_summary) CONTAINS toLower($query)
               OR toLower(t.detailed_description) CONTAINS toLower($query)
               OR toLower(t.condition) CONTAINS toLower($query)
               OR toLower(t.intervention_name) CONTAINS toLower($query)
            RETURN t.nct_id as nct_id,
                   t.title as title,
                   t.brief_summary as brief_summary,
                   t.detailed_description as detailed_description,
                   t.phase as phase,
                   t.status as status,
                   t.condition as condition,
                   t.intervention_name as intervention_name,
                   t.enrollment as enrollment,
                   t.start_date as start_date,
                   t.completion_date as completion_date
            LIMIT $limit
            """
        
        # Execute query
        parameters = {"query": query, "limit": top_k}
        raw_results = graph_db.execute_query(cypher_query, parameters)
        
        # Format results
        results = format_search_results(raw_results)
        
        response = {
            "success": True,
            "query": query,
            "search_method": "graph_database",
            "search_type": search_type,
            "num_results": len(results),
            "results": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_clinical_trials_graph: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })

@mcp.tool()
def search_trials_by_condition_graph(condition: str, top_k: int = 10) -> str:
    """
    Search for top k clinical trials with relationships to the specified condition node.
    
    Args:
        condition: The medical condition or disease name
        top_k: Number of results to return (default: 10, max: 20)
    
    Returns:
        JSON string containing trials with their relationships to the condition
    """
    try:
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database",
                "condition": condition
            })
        
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
        
        # Enhanced query to find trials with explicit relationships to condition nodes
        cypher_query = """
        MATCH (t:ClinicalTrial)-[r1:STUDY_HAS_CONDITION]-(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($condition) 
        RETURN t.id as nct_id,
               t.title as title,
               t.brief_summary as brief_summary,
               t.detailed_description as detailed_description,
               t.condition as condition,
               t.intervention_name as intervention_name,
               t.intervention_description as intervention_description,
               t.phase as phase,
               t.status as status,
               t.enrollment as enrollment,
               t.start_date as start_date,
               t.completion_date as completion_date,
               t.primary_outcome as primary_outcome,
               t.secondary_outcome as secondary_outcome
        LIMIT $limit
        """
        
        parameters = {"condition": condition, "limit": top_k}
        raw_results = graph_db.execute_query(cypher_query, parameters)
        
        # Enhanced result formatting with relationship information
        enhanced_results = []
        for i, result in enumerate(raw_results):
            enhanced_result = {
                "rank": i + 1,
                "nct_id": result.get("nct_id", ""),
                "title": result.get("title", ""),
                "brief_summary": result.get("brief_summary", ""),
                "detailed_description": result.get("detailed_description", ""),
                "condition": result.get("condition", ""),
                "intervention_name": result.get("intervention_name", ""),
                "intervention_description": result.get("intervention_description", ""),
                "phase": result.get("phase", ""),
                "status": result.get("status", ""),
                "enrollment": result.get("enrollment", ""),
                "start_date": result.get("start_date", ""),
                "completion_date": result.get("completion_date", ""),
                "primary_outcome": result.get("primary_outcome", ""),
                "secondary_outcome": result.get("secondary_outcome", ""),
                "relevance_score": result.get("relevance_score", 1),
                "relationship_info": {
                    "condition_node_name": result.get("condition_node_name"),
                    "relationship_type": result.get("relationship_type"),
                    "has_explicit_relationship": result.get("condition_node_name") is not None
                },
                "content": format_trial_content(result),
                "metadata": {
                    "search_condition": condition,
                    "nct_id": result.get("nct_id", ""),
                    "phase": result.get("phase", ""),
                    "status": result.get("status", ""),
                    "relevance_score": result.get("relevance_score", 1)
                }
            }
            enhanced_results.append(enhanced_result)
        
        # Get additional statistics about condition relationships
        stats_query = """
        MATCH (c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($condition) 
           OR toLower(c.condition) CONTAINS toLower($condition)
        OPTIONAL MATCH (c)-[r]-(t:ClinicalTrial)
        RETURN c.name as condition_name, 
               count(DISTINCT r) as relationship_count,
               count(DISTINCT t) as connected_trials,
               collect(DISTINCT type(r)) as relationship_types
        """
        
        stats_results = graph_db.execute_query(stats_query, {"condition": condition})
        
        response = {
            "success": True,
            "condition": condition,
            "search_method": "graph_database",
            "search_type": "condition_with_relationships",
            "num_results": len(enhanced_results),
            "trials": enhanced_results,
            "condition_statistics": {
                "total_trials_found": len(enhanced_results),
                "trials_with_explicit_relationships": len([r for r in enhanced_results if r["relationship_info"]["has_explicit_relationship"]]),
                "condition_nodes_found": stats_results if stats_results else [],
                "search_strategy": "Combined relationship and text-based matching"
            }
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_trials_by_condition_graph: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "condition": condition
        })

@mcp.tool()
def search_trials_by_intervention_graph(intervention: str, top_k: int = 10) -> str:
    """
    Search for clinical trials by intervention/treatment type using graph database.
    
    Args:
        intervention: The intervention, drug, or treatment name
        top_k: Number of results to return (default: 10, max: 20)
    
    Returns:
        JSON string containing trials related to the intervention
    """
    try:
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database",
                "intervention": intervention
            })
        
        if top_k > 20:
            top_k = 20
        elif top_k < 1:
            top_k = 1
        
        cypher_query = """
        MATCH (t:ClinicalTrial)-[r1:Study_has_intervention]-(i:Intervention)
        WHERE toLower(i.name) CONTAINS toLower($intervention) 
        RETURN t.id as nct_id,
               t.title as title,
               t.brief_summary as brief_summary,
               t.detailed_description as detailed_description,
               t.condition as condition,
               t.intervention_name as intervention_name,
               t.intervention_description as intervention_description,
               t.phase as phase,
               t.status as status,
               t.enrollment as enrollment,
               t.start_date as start_date,
               t.completion_date as completion_date,
               t.primary_outcome as primary_outcome,
               t.secondary_outcome as secondary_outcome
        LIMIT $limit
        """
        
        parameters = {"intervention": intervention, "limit": top_k}
        raw_results = graph_db.execute_query(cypher_query, parameters)
        results = format_search_results(raw_results)
        
        response = {
            "success": True,
            "intervention": intervention,
            "search_method": "graph_database",
            "search_type": "intervention",
            "num_results": len(results),
            "trials": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_trials_by_intervention_graph: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "intervention": intervention
        })

@mcp.tool()
def get_trial_by_nct_id_graph(nct_id: str) -> str:
    """
    Get detailed information about a specific clinical trial by NCT ID using graph database.
    
    Args:
        nct_id: The NCT ID of the clinical trial (e.g., NCT12345678)
    
    Returns:
        JSON string containing detailed trial information
    """
    try:
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database",
                "nct_id": nct_id
            })
        
        cypher_query = """
        MATCH (t:ClinicalTrial {id: $nct_id})
        RETURN t.id as nct_id,
               t.title as title,
               t.brief_summary as brief_summary,
               t.detailed_description as detailed_description,
               t.condition as condition,
               t.intervention_name as intervention_name,
               t.intervention_description as intervention_description,
               t.phase as phase,
               t.status as status,
               t.enrollment as enrollment,
               t.start_date as start_date,
               t.completion_date as completion_date,
               t.primary_outcome as primary_outcome,
               t.secondary_outcome as secondary_outcome
        """
        
        parameters = {"nct_id": nct_id}
        raw_results = graph_db.execute_query(cypher_query, parameters)
        
        if raw_results:
            trial_info = format_search_results(raw_results)[0]
            response = {
                "success": True,
                "nct_id": nct_id,
                "search_method": "graph_database",
                "found": True,
                "trial_info": trial_info
            }
        else:
            response = {
                "success": True,
                "nct_id": nct_id,
                "search_method": "graph_database",
                "found": False,
                "trial_info": None
            }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_trial_by_nct_id_graph: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "nct_id": nct_id
        })

@mcp.tool()
def get_trial_answer_graph(query: str, top_k: int = 5, llm_provider: str = "deepseek", model_name: Optional[str] = None) -> str:
    """
    Get an AI-generated answer about clinical trials using graph-based RAG.
    
    Args:
        query: The question about clinical trials
        top_k: Number of documents to retrieve for context (default: 5, max: 10)
        llm_provider: LLM provider to use for answer generation
        model_name: Specific model name (optional)
    
    Returns:
        JSON string containing the generated answer and source documents
    """
    try:
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database",
                "query": query
            })
        
        if top_k > 10:
            top_k = 10
        elif top_k < 1:
            top_k = 1
        
        # First, search for relevant documents using the graph search function
        search_result = search_clinical_trials_graph(query, top_k, "auto")
        search_data = json.loads(search_result)
        
        if not search_data.get("success"):
            return search_result  # Return the error from search
        
        context_docs = search_data.get("results", [])
        
        if not context_docs:
            return json.dumps({
                "success": True,
                "query": query,
                "search_method": "graph_database",
                "answer": "No relevant clinical trials found for your query.",
                "source_documents": [],
                "llm_provider": llm_provider,
                "model_name": model_name
            })
        
        # Generate answer using LLM
        llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=0.2)
        
        # Format context from documents
        context = "\n\n".join([
            f"Clinical Trial {doc['rank']}:\n"
            f"NCT ID: {doc.get('nct_id', 'N/A')}\n"
            f"Title: {doc.get('title', 'N/A')}\n"
            f"Summary: {doc.get('brief_summary', 'N/A')}\n"
            f"Condition: {doc.get('condition', 'N/A')}\n"
            f"Intervention: {doc.get('intervention_name', 'N/A')}\n"
            f"Phase: {doc.get('phase', 'N/A')}\n"
            f"Status: {doc.get('status', 'N/A')}"
            for doc in context_docs
        ])
        
        # Create prompt
        prompt = f"""Based on the following clinical trials information from a graph database, answer the user's question.

Context from Clinical Trials Graph Database:
{context}

Question: {query}

Please provide a comprehensive answer based on the clinical trials data above. Include relevant NCT IDs and specific trial details when applicable. If the information is not sufficient to answer the question completely, indicate what additional information might be needed.

Answer:"""
        
        # Generate response
        response = llm.invoke(prompt)
        
        # Extract content based on response type
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        response_data = {
            "success": True,
            "query": query,
            "search_method": "graph_database",
            "answer": answer,
            "source_documents": context_docs,
            "num_source_documents": len(context_docs),
            "llm_provider": llm_provider,
            "model_name": model_name or "default"
        }
        
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_trial_answer_graph: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })

# =============================================================================
# SYSTEM TOOLS
# =============================================================================

@mcp.tool()
def get_database_stats() -> str:
    """
    Get statistics about the graph database.
    
    Returns:
        JSON string containing database statistics
    """
    try:
        if not initialize_graph_connection():
            return json.dumps({
                "success": False,
                "error": "Failed to connect to graph database"
            })
        
        stats = graph_db.get_database_stats()
        
        response = {
            "success": True,
            "database_stats": stats,
            "connection_status": "connected"
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_database_stats: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@mcp.tool()
def get_rag_system_status() -> str:
    """
    Get the current status of both RAG systems (vector and graph).
    
    Returns:
        JSON string containing system status information
    """
    try:
        # Check vector system status
        vector_status = {
            "initialized": vector_rag_system.initialized,
            "embedding_provider": vector_rag_system.embedding_provider,
            "vectorstore_available": vector_rag_system.vectorstore is not None,
            "retriever_available": vector_rag_system.retriever is not None
        }
        
        if vector_rag_system.initialized and vector_rag_system.vectorstore:
            try:
                if hasattr(vector_rag_system.vectorstore, '_collection'):
                    vector_status["vectorstore_type"] = type(vector_rag_system.vectorstore).__name__
                vector_status["system_ready"] = True
            except:
                vector_status["system_ready"] = False
        else:
            vector_status["system_ready"] = False
        
        # Check graph system status  
        graph_status = {
            "connected": graph_db.connected,
            "config": {
                "uri": graph_db.config.get("dburi", "unknown"),
                "database": graph_db.config.get("database", "neo4j")
            },
            "system_ready": graph_db.connected
        }
        
        # Get database stats if connected
        if graph_db.connected:
            try:
                stats = graph_db.get_database_stats()
                graph_status["database_stats"] = stats
                graph_status["has_data"] = stats.get("total_nodes", 0) > 0
            except Exception as e:
                graph_status["database_stats"] = {"error": str(e)}
                graph_status["has_data"] = False
        
        status_info = {
            "success": True,
            "vector_rag_system": vector_status,
            "graph_rag_system": graph_status,
            "overall_status": {
                "vector_ready": vector_status.get("system_ready", False),
                "graph_ready": graph_status.get("system_ready", False),
                "any_system_ready": vector_status.get("system_ready", False) or graph_status.get("system_ready", False),
                "both_systems_ready": vector_status.get("system_ready", False) and graph_status.get("system_ready", False)
            }
        }
        
        return json.dumps(status_info, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_rag_system_status: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

def main():
    """Main function to run the unified RAG MCP server"""
    logger.info("Starting Unified Clinical Trials RAG MCP Server...")
    logger.info("This server provides both vector-based and graph-based RAG capabilities")
    
    # Pre-initialize both systems
    logger.info("Pre-initializing vector RAG system...")
    if initialize_vector_system():
        logger.info("Vector RAG system ready!")
    else:
        logger.warning("Vector RAG system initialization failed. Will retry on first request.")
    
    logger.info("Pre-initializing graph database connection...")
    if initialize_graph_connection():
        logger.info("Graph database connection ready!")
        
        # Get initial stats
        try:
            stats = graph_db.get_database_stats()
            logger.info(f"Database stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")
    else:
        logger.warning("Graph database connection failed. Will retry on first request.")
    
    # Run the MCP server
    logger.info("Starting unified MCP server on http://127.0.0.1:8000/mcp")
    logger.info("Available tools:")
    logger.info("  Vector-based: search_clinical_trials, get_trial_answer, get_trial_by_nct_id, search_trials_by_condition_vector")
    logger.info("  Graph-based: search_clinical_trials_graph, get_trial_answer_graph, get_trial_by_nct_id_graph, search_trials_by_condition_graph, search_trials_by_intervention_graph") 
    logger.info("  System: get_rag_system_status, get_database_stats")
    
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")

if __name__ == "__main__":
    main() 