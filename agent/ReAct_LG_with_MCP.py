#!/usr/bin/env python3
"""
ReAct_LG_with_MCP.py - ReAct agent implementation using LangGraph with MCP RAG tool

This version uses an external RAG MCP server as a tool instead of built-in RAG functionality.
The agent only checks MCP server availability and does not start it (for docker-compose deployment).
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Optional
import argparse
from pydantic import BaseModel, Field

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
from RAG.rag_utils import get_llm_model

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

class MCPRAGTool(BaseTool):
    """Custom tool that communicates with the RAG MCP server."""
    
    name: str = "ClinicalTrialsRAG"
    description: str = "Search for clinical trials information using advanced RAG capabilities. Use this for questions about specific clinical trials, medical treatments, patient criteria, or trial outcomes."
    mcp_server_url: str = "http://localhost:8000"  # Default MCP server URL
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000", **kwargs):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
        
    def _run(self, query: str, search_type: str = "search", **kwargs) -> str:
        """Execute the tool by communicating with the MCP server."""
        try:
            if search_type == "answer":
                # Use get_trial_answer for comprehensive answers
                payload = {
                    "name": "get_trial_answer",
                    "arguments": {
                        "query": query,
                        "top_k": kwargs.get("top_k", 5),
                        "llm_provider": kwargs.get("llm_provider", "mistral")
                    }
                }
            else:
                # Use search_clinical_trials for document retrieval
                payload = {
                    "name": "search_clinical_trials", 
                    "arguments": {
                        "query": query,
                        "top_k": kwargs.get("top_k", 5)
                    }
                }
            
            response = requests.post(
                f"{self.mcp_server_url}/tools",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.dumps(result, indent=2)
            else:
                return f"Error: MCP server responded with status {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error communicating with RAG MCP server: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _arun(self, query: str, **kwargs) -> str:
        """Async version - not implemented for this example."""
        return self._run(query, **kwargs)

def create_mcp_rag_tools(mcp_server_url: str = "http://localhost:8000/mcp") -> List[BaseTool]:
    """Create MCP RAG tools for different search types."""
    tools = []
    
    # Main RAG search tool
    main_rag_tool = MCPRAGTool(
        mcp_server_url=mcp_server_url,
        name="ClinicalTrialsRAG",
        description="Search clinical trials database using advanced semantic search. Returns relevant trial documents and metadata."
    )
    tools.append(main_rag_tool)
    
    # RAG answer tool
    answer_rag_tool = Tool(
        name="ClinicalTrialsAnswer",
        func=lambda query: main_rag_tool._run(query, search_type="answer"),
        description="Get comprehensive AI-generated answers about clinical trials using RAG. Use this when you need detailed explanations rather than just document retrieval."
    )
    tools.append(answer_rag_tool)
    
    # Specific NCT ID lookup
    nct_lookup_tool = Tool(
        name="NCTLookup",
        func=lambda nct_id: requests.post(
            f"{mcp_server_url}/tools",
            json={"name": "get_trial_by_nct_id", "arguments": {"nct_id": nct_id}},
            timeout=30
        ).json() if requests.post(
            f"{mcp_server_url}/tools",
            json={"name": "get_trial_by_nct_id", "arguments": {"nct_id": nct_id}},
            timeout=30
        ).status_code == 200 else f"Error looking up NCT ID {nct_id}",
        description="Look up specific clinical trial by NCT ID (e.g., NCT12345678). Use this when you have a specific trial identifier."
    )
    tools.append(nct_lookup_tool)
    
    return tools

def create_search_tools() -> List[BaseTool]:
    """Create a set of internet search tools."""
    tools = []
    
    # Add DuckDuckGo Search
    try:
        ddg_search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Search the web for current information, news, or recent events. Use this for information not available in the clinical trials database."
            )
        )
        print("Added DuckDuckGo search tool")
    except ImportError:
        print("Could not import DuckDuckGo search.")
    
    # Add Wikipedia Search
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(
            Tool(
                name="WikipediaSearch",
                func=wikipedia.run,
                description="Search Wikipedia for factual information and general medical knowledge."
            )
        )
        print("Added Wikipedia search tool")
    except ImportError:
        print("Could not import Wikipedia tool.")
    
    # Add ArXiv Search for scientific papers
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        tools.append(
            Tool(
                name="ArXivSearch",
                func=arxiv.run,
                description="Search scientific papers on ArXiv for research about clinical trials and medical treatments."
            )
        )
        print("Added ArXiv search tool")
    except ImportError:
        print("Could not import ArXiv tool.")
        
    return tools

def check_mcp_server_status(mcp_server_url: str) -> bool:
    """Check if the MCP RAG server is running."""
    try:
        response = requests.post(
            f"{mcp_server_url}/tools",
            json={"name": "get_rag_system_status", "arguments": {}},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("success", False)
        return False
    except:
        return False

def create_agent_graph(llm_provider="mistral", model_name=None, temperature=0.2, 
                      mcp_server_url="http://localhost:8000"):
    """Create a LangGraph-based ReAct agent with MCP RAG capabilities."""
    
    # Check if MCP server is running (do not start it)
    if not check_mcp_server_status(mcp_server_url):
        print(f"Warning: RAG MCP server at {mcp_server_url} is not available.")
        print("Please ensure the RAG MCP server is running before using clinical trials tools.")
        print("The agent will still work with internet search tools only.")
    else:
        print(f"RAG MCP server at {mcp_server_url} is available.")
    
    # Get LLM
    llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Create tools
    rag_tools = create_mcp_rag_tools(mcp_server_url)
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
    
    def clinical_search(state: AgentState) -> Dict:
        """Search using RAG MCP tools."""
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
                        result = tool.invoke(tool_call["args"])
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
        
        return {
            "messages": [response],
            "clinical_context": clinical_results,
            "current_task": "internet_search"
        }
    
    def internet_search(state: AgentState) -> Dict:
        """Search the internet for supplementary information."""
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
                        result = tool.invoke(tool_call["args"])
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
        
        return {
            "messages": [response],
            "internet_context": state["internet_context"] + internet_results,
            "current_task": "final_answer",
            "all_searches_completed": True
        }
    
    def formulate_final_answer(state: AgentState) -> Dict:
        """Synthesize information to generate final answer."""
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
    parser.add_argument("--mcp-server-url", default="http://localhost:8000",
                      help="URL of the RAG MCP server")
    return parser.parse_args()

def main():
    """Main function to run the ReAct agent with MCP RAG."""
    args = parse_arguments()
    
    # Create the agent graph
    agent_graph = create_agent_graph(
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
            
            final_state = agent_graph.invoke(
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
    main() 