#!/usr/bin/env python3
"""
ReAct_LG_with_MCP.py - ReAct agent using Official LangChain MCP Adapters

This version uses an external RAG MCP server as a tool instead of built-in RAG functionality.
The agent only checks MCP server availability and does not start it (for docker-compose deployment).
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Optional
import argparse
from pydantic import BaseModel, Field

# Official LangChain MCP Adapters - THE SOLUTION!
from langchain_mcp_adapters.client import MultiServerMCPClient

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import Tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only LLM functionality from RAG
from llm_utils import get_llm_model

# Define the graph state
class AgentState(TypedDict):
    """State for the ReAct agent."""
    # Messages channel - contains the conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw content from the clinical trials RAG
    clinical_context: List[str]
    # Raw content from internet searches
    internet_context: List[str]
    # Current task state
    current_task: str
    # Current question being processed
    question: str
    # Flag to indicate if all searches have been completed
    all_searches_completed: bool

# MCPRAGTool class removed - now using direct tool exposure for better LLM decision making

async def create_mcp_rag_tools(mcp_server_url: str = "http://127.0.0.1:8000/mcp") -> List[BaseTool]:
    """
    Create MCP RAG tools using Official LangChain MCP Adapters.
    
    """
    try:
        # Create MCP client using official adapter
        # Note: HTTP transport is an alias for streamable_http in the official adapter
        client = MultiServerMCPClient({
            "clinical_trials_rag": {
                "url": mcp_server_url,
                "transport": "streamable_http"  # Your server supports this via HTTP alias
            }
        })
        
        # Get tools using official method - NO CUSTOM WRAPPERS!
        print("🚀 Loading tools using Official LangChain MCP Adapters...")
        tools = await client.get_tools()
        
        print(f"✅ Successfully loaded {len(tools)} MCP tools using official adapter!")
        for i, tool in enumerate(tools, 1):
            print(f"  {i:2d}. {tool.name}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Failed to load MCP tools using official adapter: {e}")
        print("⚠️  Falling back to empty tools list. Make sure MCP server is running.")
        return []

def create_search_tools() -> List[BaseTool]:
    """Create a set of internet search tools."""
    tools = []
    
    # Add DuckDuckGo Search with error handling
    try:
        ddg_search = DuckDuckGoSearchRun()
        
        def safe_duckduckgo_search(query: str) -> str:
            """DuckDuckGo search with error handling for network and API issues."""
            try:
                result = ddg_search.run(query)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "tls handshake failed" in error_msg or "connection reset" in error_msg:
                    return f"DuckDuckGo search temporarily unavailable due to network connectivity issues. Query: {query}"
                elif "client error" in error_msg or "connect" in error_msg:
                    return f"DuckDuckGo search service temporarily unreachable. Query: {query}"
                elif "timeout" in error_msg:
                    return f"DuckDuckGo search timed out. Query: {query}"
                elif "rate limit" in error_msg or "too many requests" in error_msg:
                    return f"DuckDuckGo search rate limited. Please try again later. Query: {query}"
                else:
                    return f"DuckDuckGo search error for '{query}': {str(e)}"
        
        # tools.append(
        #     Tool(
        #         name="DuckDuckGoSearch",
        #         func=safe_duckduckgo_search,
        #         description="Search the web for current information, news, or recent events. Use this for information not available in the clinical trials database."
        #     )
        # )
        # print("Added DuckDuckGo search tool with error handling")
    except ImportError:
        print("Could not import DuckDuckGo search.")
    
    # Add Wikipedia Search with error handling
    try:
        wikipedia_api = WikipediaAPIWrapper()
        wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api)
        
        def safe_wikipedia_search(query: str) -> str:
            """Wikipedia search with error handling for JSON decode issues."""
            try:
                result = wikipedia.run(query)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "json" in error_msg or "expecting value" in error_msg:
                    return f"Wikipedia search temporarily unavailable (API issue). Query: {query}"
                elif "disambiguation" in error_msg:
                    return f"Wikipedia disambiguation found for '{query}'. Please be more specific."
                elif "page" in error_msg and "does not exist" in error_msg:
                    return f"No Wikipedia page found for '{query}'."
                else:
                    return f"Wikipedia search error for '{query}': {str(e)}"
        
        tools.append(
            Tool(
                name="WikipediaSearch",
                func=safe_wikipedia_search,
                description="Search Wikipedia for factual information and general medical knowledge."
            )
        )
        print("Added Wikipedia search tool with error handling")
    except ImportError:
        print("Could not import Wikipedia tool.")
    
    # Add ArXiv Search for scientific papers with error handling
    try:
        arxiv_api = ArxivAPIWrapper()
        arxiv = ArxivQueryRun(api_wrapper=arxiv_api)
        
        def safe_arxiv_search(query: str) -> str:
            """ArXiv search with error handling for network and API issues."""
            try:
                result = arxiv.run(query)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                    return f"ArXiv search temporarily unavailable due to network issues. Query: {query}"
                elif "no papers found" in error_msg or "no results" in error_msg:
                    return f"No ArXiv papers found for '{query}'. Try broader search terms."
                else:
                    return f"ArXiv search error for '{query}': {str(e)}"
        
        tools.append(
            Tool(
                name="ArXivSearch",
                func=safe_arxiv_search,
                description="Search scientific papers on ArXiv for research about clinical trials and medical treatments."
            )
        )
        print("Added ArXiv search tool with error handling")
    except ImportError:
        print("Could not import ArXiv tool.")
        
    return tools

async def check_mcp_server_status(mcp_server_url: str) -> bool:
    """Check if the MCP RAG server is running using official adapter."""
    try:
        # Use official adapter for connection test
        client = MultiServerMCPClient({
            "clinical_trials_rag": {
                "url": mcp_server_url,
                "transport": "streamable_http"
            }
        })
        
        # Try to get tools list as a connection test
        tools = await client.get_tools()
        return len(tools) > 0
        
    except Exception:
        return False

async def create_agent_graph(llm_provider="mistral", model_name=None, temperature=0.2, 
                      mcp_server_url="http://127.0.0.1:8000/mcp"):
    """Create a LangGraph-based ReAct agent with Official MCP Adapters."""
    
    # Check if MCP server is running using official adapter
    server_available = await check_mcp_server_status(mcp_server_url)
    if not server_available:
        print(f"Warning: RAG MCP server at {mcp_server_url} is not available.")
        print("Please ensure the RAG MCP server is running before using clinical trials tools.")
        print("The agent will still work with internet search tools only.")
    else:
        print(f"✅ RAG MCP server at {mcp_server_url} is available (verified with official adapter)")
    
    # Get LLM
    llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Create tools using official adapter
    rag_tools = await create_mcp_rag_tools(mcp_server_url)
    internet_tools = create_search_tools()
    all_tools = rag_tools + internet_tools
    
    print(f"Created {len(all_tools)} tools: {[tool.name for tool in all_tools]}")
    
    # Create system prompts
    core_system_prompt = """You are a clinical trials expert assistant with access to both a specialized clinical trials database and internet search capabilities.

To answer questions effectively:
1. First search the clinical trials database using RAG tools for specific trial information
2. Supplement with internet searches for additional context or recent developments
3. Combine information from both sources to provide comprehensive answers

Always cite your sources and indicate which information came from the clinical trials database versus internet sources.
"""

    clinical_search_prompt = core_system_prompt + """
CURRENT TASK: CLINICAL DATABASE SEARCH

Use the available RAG tools to search the clinical trials database:
- ClinicalTrialsRAG: For document retrieval and semantic search
- ClinicalTrialsAnswer: For AI-generated comprehensive answers
- NCTLookup: For specific NCT ID lookups

Choose the most appropriate tool based on the query type.
"""

    internet_search_prompt = core_system_prompt + """
CURRENT TASK: INTERNET SEARCH

Supplement clinical trials information with internet searches using available tools:
- DuckDuckGoSearch: For current web information
- WikipediaSearch: For general medical knowledge
- ArXivSearch: For scientific papers

Use these to find additional context, recent developments, or general medical information.
"""

    final_answer_prompt = core_system_prompt + """
CURRENT TASK: FINAL ANSWER

Synthesize information from all sources into a comprehensive answer.

Clinical Database Information:
{clinical_context}

Internet Search Information:
{internet_context}

Provide a complete answer that integrates both sources, clearly citing where each piece of information originated.
"""

    # Define graph nodes
    def process_question(state: AgentState) -> Dict:
        """Process the user question and set up initial state."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
        
        return {
            "current_task": "clinical_search",
            "question": question,
            "clinical_context": [],
            "internet_context": [],
            "all_searches_completed": False
        }
    
    async def clinical_search(state: AgentState) -> Dict:
        """Search using RAG MCP tools."""
        print("\n📊 Phase 1: Searching Clinical Trials Database...")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=clinical_search_prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Question: {question}\n\nSearch the clinical trials database for relevant information."),
        ])
        
        # Bind RAG tools to the LLM
        clinical_llm = llm.bind_tools(rag_tools)
        
        messages = prompt.format(messages=state["messages"], question=state["question"])
        response = clinical_llm.invoke(messages)
        
        # Process tool calls
        clinical_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                for tool in rag_tools:
                    if tool_call["name"] == tool.name:
                        try:
                            # Show user which tool is being called
                            print(f"🔍 Calling Clinical Database Tool: {tool.name}")
                            print(f"   └── Arguments: {tool_call['args']}")
                            
                            # Check if tool is async by checking for ainvoke method
                            if hasattr(tool, 'ainvoke'):
                                result = await tool.ainvoke(tool_call["args"])
                            else:
                                result = tool.invoke(tool_call["args"])
                            
                            print(f"   ✅ Tool completed successfully")
                            clinical_results.append(f"{tool.name}: {result}")
                            
                            tool_msg = ToolMessage(
                                content=str(result),
                                name=tool.name,
                                tool_call_id=tool_call["id"]
                            )
                            return {
                                "messages": [response, tool_msg],
                                "clinical_context": clinical_results,
                                "current_task": "internet_search"
                            }
                        except Exception as e:
                            error_result = f"Error using {tool.name}: {str(e)}"
                            clinical_results.append(f"{tool.name}: {error_result}")
                            print(f"   ❌ Clinical tool error: {error_result}")
                            
                            tool_msg = ToolMessage(
                                content=error_result,
                                name=tool.name,
                                tool_call_id=tool_call["id"]
                            )
                            return {
                                "messages": [response, tool_msg],
                                "clinical_context": clinical_results,
                                "current_task": "internet_search"
                            }
        
        if not clinical_results:
            print("   ℹ️  No clinical database tools were called for this query")
        
        return {
            "messages": [response],
            "clinical_context": clinical_results,
            "current_task": "internet_search"
        }
    
    async def internet_search(state: AgentState) -> Dict:
        """Search the internet for supplementary information."""
        print("\n🌐 Phase 2: Searching Internet for Additional Information...")
        
        internet_tool_names = ", ".join([tool.name for tool in internet_tools])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=internet_search_prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Question: {question}\n\nClinical Info: {clinical_info}\n\nSearch the internet for additional information."),
        ])
        
        internet_llm = llm.bind_tools(internet_tools)
        
        clinical_info = "\n".join(state["clinical_context"]) if state["clinical_context"] else "No clinical trial information found."
        messages = prompt.format(
            messages=state["messages"], 
            question=state["question"],
            clinical_info=clinical_info
        )
        response = internet_llm.invoke(messages)
        
        # Process tool calls
        internet_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                for tool in internet_tools:
                    if tool_call["name"] == tool.name:
                        try:
                            # Show user which tool is being called
                            print(f"🌐 Calling Internet Search Tool: {tool.name}")
                            print(f"   └── Arguments: {tool_call['args']}")
                            
                            # Check if tool is async by checking for ainvoke method
                            if hasattr(tool, 'ainvoke'):
                                result = await tool.ainvoke(tool_call["args"])
                            else:
                                result = tool.invoke(tool_call["args"])
                            
                            print(f"   ✅ Tool completed successfully")
                            internet_results.append(f"{tool.name}: {result}")
                            
                            tool_msg = ToolMessage(
                                content=str(result),
                                name=tool.name,
                                tool_call_id=tool_call["id"]
                            )
                            return {
                                "messages": [response, tool_msg],
                                "internet_context": state["internet_context"] + internet_results,
                                "current_task": "final_answer",
                                "all_searches_completed": True
                            }
                        except Exception as e:
                            error_result = f"Error using {tool.name}: {str(e)}"
                            internet_results.append(f"{tool.name}: {error_result}")
                            print(f"   ❌ Internet tool error: {error_result}")
                            
                            tool_msg = ToolMessage(
                                content=error_result,
                                name=tool.name,
                                tool_call_id=tool_call["id"]
                            )
                            return {
                                "messages": [response, tool_msg],
                                "internet_context": state["internet_context"] + internet_results,
                                "current_task": "final_answer",
                                "all_searches_completed": True
                            }
        
        if not internet_results:
            print("   ℹ️  No internet search tools were called for this query")
        
        return {
            "messages": [response],
            "internet_context": state["internet_context"] + internet_results,
            "current_task": "final_answer",
            "all_searches_completed": True
        }
    
    def formulate_final_answer(state: AgentState) -> Dict:
        """Synthesize information to generate final answer."""
        print("\n🤖 Phase 3: Synthesizing Information and Generating Final Answer...")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=final_answer_prompt.format(
                clinical_context="\n".join(state["clinical_context"]) if state["clinical_context"] else "No clinical trial information found.",
                internet_context="\n".join(state["internet_context"]) if state["internet_context"] else "No internet information found."
            )),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Provide a comprehensive answer to: {question}"),
        ])
        
        messages = prompt.format(messages=state["messages"], question=state["question"])
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "current_task": "complete"
        }
    
    def router(state: AgentState) -> str:
        """Route the flow based on current task state."""
        if state["current_task"] == "clinical_search":
            return "clinical_search"
        elif state["current_task"] == "internet_search":
            return "internet_search"
        elif state["current_task"] == "final_answer" or state["all_searches_completed"]:
            return "final_answer"
        else:
            return "end"
    
    # Create and configure the graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("process_question", process_question)
    workflow.add_node("clinical_search", clinical_search)
    workflow.add_node("internet_search", internet_search)
    workflow.add_node("final_answer", formulate_final_answer)
    
    workflow.set_entry_point("process_question")
    
    workflow.add_conditional_edges(
        "process_question",
        router,
        {
            "clinical_search": "clinical_search",
            "internet_search": "internet_search", 
            "final_answer": "final_answer",
            "end": END,
        },
    )
    
    workflow.add_conditional_edges(
        "clinical_search",
        router,
        {
            "clinical_search": "clinical_search",
            "internet_search": "internet_search",
            "final_answer": "final_answer",
            "end": END,
        },
    )
    
    workflow.add_conditional_edges(
        "internet_search",
        router,
        {
            "clinical_search": "clinical_search",
            "internet_search": "internet_search",
            "final_answer": "final_answer",
            "end": END,
        },
    )
    
    workflow.add_edge("final_answer", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clinical Trials ReAct Agent with MCP RAG")
    parser.add_argument("--llm", default="deepseek", 
                      choices=["openai", "anthropic", "huggingface", "mistral", "gemini","deepseek"],
                      help="LLM provider to use")
    parser.add_argument("--model", default="deepseek-chat", 
                      help="Specific model name")
    parser.add_argument("--temperature", type=float, default=0.2, 
                      help="Model temperature (0-1)")
    parser.add_argument("--mcp-server-url", default="http://127.0.0.1:8000/mcp",
                      help="URL of the RAG MCP server")
    return parser.parse_args()

async def main():
    """Main function to run the ReAct agent with Official MCP Adapters."""
    args = parse_arguments()
    
    # Create the agent graph using official adapters
    agent_graph = await create_agent_graph(
        llm_provider=args.llm,
        model_name=args.model,
        temperature=args.temperature,
        mcp_server_url=args.mcp_server_url
    )
    
    if not agent_graph:
        print("Failed to create agent. Exiting.")
        return
    
    print("\n=== Clinical Trials LangGraph ReAct Agent with MCP RAG ===")
    print(f"Using LLM: {args.llm}{f' ({args.model})' if args.model else ''}")
    print(f"RAG MCP Server: {args.mcp_server_url}")
    print("Type 'exit' to quit")
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
            
        try:
            input_state = {
                "messages": [HumanMessage(content=query)],
                "clinical_context": [],
                "internet_context": [],
                "current_task": "start",
                "question": query,
                "all_searches_completed": False
            }
            
            print("\nProcessing your question...")
            
            final_state = await agent_graph.ainvoke(
                input_state,
                config={"configurable": {"thread_id": "default_thread"}}
            )
            
            final_message = final_state["messages"][-1]
            print("\nAnswer:", final_message.content)
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 