# ReAct_LG.py - ReAct agent implementation using LangGraph

import os
import sys
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Tuple, Union, Literal
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
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from RAG directory
from RAG.rag_utils import get_llm_model, get_embedding_model, setup_retriever, fetch_trials_from_mongo, transform_trials_to_documents, create_vectorstore

# Define the graph state
class AgentState(TypedDict):
    """State for the ReAct agent."""
    # Messages channel - contains the conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw content from the clinical trials DB
    clinical_context: List[str]
    # Raw content from internet searches
    internet_context: List[str]
    # Current task state (can be: start, clinical_search, internet_search, final_answer)
    current_task: str
    # Current question being processed
    question: str
    # Flag to indicate if all searches have been completed
    all_searches_completed: bool

def create_search_tools() -> List[BaseTool]:
    """Create a set of internet search tools.
    
    Returns:
        List of search tools
    """
    tools = []
    
    # Add DuckDuckGo Search (no API key required)
    try:
        ddg_search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for searching the web for current information, news, or recent events that might not be in the knowledge base. Input should be a search query."
            )
        )
        print("Added DuckDuckGo search tool")
    except ImportError:
        print("Could not import DuckDuckGo search. Install with: pip install duckduckgo-search")
    
    # Add Wikipedia Search (no API key required)
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(
            Tool(
                name="WikipediaSearch",
                func=wikipedia.run,
                description="Useful for searching for factual information, historical data, or well-established concepts. Input should be a search query."
            )
        )
        print("Added Wikipedia search tool")
    except ImportError:
        print("Could not import Wikipedia tool. Install with: pip install wikipedia")
    
    # Add ArXiv Search for scientific papers (no API key required)
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        tools.append(
            Tool(
                name="ArXivSearch",
                func=arxiv.run,
                description="Useful for finding scientific papers and research about clinical trials, medical treatments, and related science. Input should be a search query."
            )
        )
        print("Added ArXiv search tool")
    except ImportError:
        print("Could not import ArXiv tool. Install with: pip install arxiv")
        
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily_search = TavilySearchResults(max_results=5)
            tools.append(
                Tool(
                    name="TavilySearch",
                    func=tavily_search.run,
                    description="A more powerful search engine that provides structured, high-quality information. Use this for complex queries requiring accurate and up-to-date information."
                )
            )
            print("Added Tavily search tool")
        except ImportError:
            print("Could not import Tavily search. Install with: pip install tavily-python")
    
    return tools

def create_agent_graph(llm_provider="mistral", model_name=None, temperature=0.2, embedding_provider="huggingface"):
    """Create a LangGraph-based ReAct agent with RAG capabilities and internet search.
    
    Args:
        llm_provider: The LLM provider to use
        model_name: Specific model name
        temperature: Temperature for generation
        embedding_provider: Provider for embeddings
        
    Returns:
        A compiled StateGraph for the ReAct agent
    """
    # Get LLM
    llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Get or create vectorstore for RAG
    print("Setting up clinical trials knowledge base...")
    trials = fetch_trials_from_mongo()
    if not trials:
        print("No trials found in MongoDB. Please check your database connection.")
        return None
    
    documents = transform_trials_to_documents(trials, embedding_provider=embedding_provider )
    vectorstore = create_vectorstore(documents, embedding_provider=embedding_provider)
    if not vectorstore:
        print("Failed to create vector store. Please check your embedding configuration.")
        return None
    
    # Set up retriever
    retriever = setup_retriever(vectorstore, embedding_provider=embedding_provider)
    
    # Create retriever tool for clinical trials database
    retriever_tool = create_retriever_tool(
        retriever,
        name="ClinicalTrialsDB",
        description="Search for information about clinical trials in the database. Use this for questions about specific clinical trials, medical treatments being studied, or patient eligibility criteria."
    )
    
    # Get internet search tools
    internet_tools = create_search_tools()
    
    # Combine all tools
    all_tools = [retriever_tool] + internet_tools
    
    # Create system prompts for different stages
    core_system_prompt = """You are a clinical trials expert assistant that can search both a clinical trials database and the internet.
    
To answer questions effectively:
1. First search the clinical trials database for specific trial information
2. Then supplement with internet searches for additional context or recent developments
3. Combine information from both sources to provide a comprehensive answer

Always cite your sources clearly, indicating which information came from the clinical trials database versus internet sources.
Keep responses concise, factual and evidence-based.
"""

    clinical_search_prompt = core_system_prompt + """
CURRENT TASK: CLINICAL DATABASE SEARCH

You are currently in the clinical database search phase. Your goal is to retrieve relevant information from the clinical trials database.

You can use the ClinicalTrialsDB tool to search for information. Formulate your search query to find the most relevant clinical trial information with vector similarity.
"""

    internet_search_prompt = core_system_prompt + """
CURRENT TASK: INTERNET SEARCH

You are currently in the internet search phase. Your goal is to supplement the clinical trials information with relevant data from the internet.

You can use any of the following search tools: {internet_tools}. Choose the most appropriate tool for finding additional, up-to-date information to complement what we already know from the clinical database.
"""

    final_answer_prompt = core_system_prompt + """
CURRENT TASK: FINAL ANSWER FORMULATION

You have gathered information from both the clinical trials database and internet sources. Now synthesize this information into a comprehensive answer.

Clinical database information:
{clinical_context}

Internet search information:
{internet_context}

Provide a complete answer that combines both sources of information, clearly citing where each piece of information came from.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one or many of {all_tools}, and specify how you use the tool
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, list citations for each source
"""

    # Define graph nodes
    
    # 1. Process question and set up initial state
    def process_question(state: AgentState) -> Dict:
        """Process the user question and set up the initial state."""
        # Extract the question from the last human message
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
    
    # 2. Perform clinical database search
    def clinical_search(state: AgentState) -> Dict:
        """Search the clinical trials database."""
        # Create the clinical search prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=clinical_search_prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Question: {question}\n\nSearch the clinical trials database for relevant information."),
        ])
        
        # Bind the ClinicalTrialsDB tool to the LLM
        clinical_llm = llm.bind_tools([retriever_tool])
        
        # Generate the response using the tool
        messages = prompt.format(messages=state["messages"], question=state["question"])
        response = clinical_llm.invoke(messages)
        
        # Extract any results from tool calls
        clinical_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "ClinicalTrialsDB":
                    # Execute the tool call
                    result = retriever_tool.invoke(tool_call["args"])
                    clinical_results.append(result)
                    
                    # Add a tool message with the result
                    tool_msg = ToolMessage(
                        content=str(result),
                        name="ClinicalTrialsDB",
                        tool_call_id=tool_call["id"]
                    )
                    return {
                        "messages": [response, tool_msg],
                        "clinical_context": clinical_results,
                        "current_task": "internet_search"
                    }
        
        # If no tool calls, just return the response and move to internet search
        return {
            "messages": [response],
            "clinical_context": clinical_results,
            "current_task": "internet_search"
        }
    
    # 3. Perform internet search
    def internet_search(state: AgentState) -> Dict:
        """Search the internet for supplementary information."""
        # Format the list of internet tools
        internet_tool_names = ", ".join([tool.name for tool in internet_tools])
        
        # Create the internet search prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=internet_search_prompt.format(internet_tools=internet_tool_names)),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Question: {question}\n\nClinical Database Info: {clinical_info}\n\nNow search the internet for additional relevant information."),
        ])
        
        # Bind the internet tools to the LLM
        internet_llm = llm.bind_tools(internet_tools)
        
        # Generate the response using the tools
        clinical_info = "\n".join(state["clinical_context"]) if state["clinical_context"] else "No specific clinical trial information found."
        messages = prompt.format(
            messages=state["messages"], 
            question=state["question"],
            clinical_info=clinical_info
        )
        response = internet_llm.invoke(messages)
        
        # Extract any results from tool calls
        internet_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                for tool in internet_tools:
                    if tool_call["name"] == tool.name:
                        # Execute the tool call
                        result = tool.invoke(tool_call["args"])
                        internet_results.append(f"{tool.name}: {result}")
                        
                        # Add a tool message with the result
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
        
        # If no tool calls, just return the response and move to final answer
        return {
            "messages": [response],
            "internet_context": state["internet_context"] + internet_results,
            "current_task": "final_answer",
            "all_searches_completed": True
        }
    
    # 4. Formulate final answer
    def formulate_final_answer(state: AgentState) -> Dict:
        """Synthesize information from both sources to generate a final answer."""
        # Create the final answer prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=final_answer_prompt.format(
                clinical_context="\n".join(state["clinical_context"]) if state["clinical_context"] else "No specific clinical trial information found.",
                internet_context="\n".join(state["internet_context"]) if state["internet_context"] else "No additional internet information found.",
                all_tools=", ".join([tool.name for tool in all_tools])
            )),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="Based on all the information gathered, provide a comprehensive answer to the original question: {question}"),
        ])
        
        # Generate the final response
        messages = prompt.format(messages=state["messages"], question=state["question"])
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "current_task": "complete"
        }
    
    # Define the routing logic
    def router(state: AgentState) -> str:
        """Route the flow based on the current task state."""
        if state["current_task"] == "clinical_search":
            return "clinical_search"
        elif state["current_task"] == "internet_search":
            return "internet_search"
        elif state["current_task"] == "final_answer" or state["all_searches_completed"]:
            return "final_answer"
        else:
            return "end"
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_question", process_question)
    workflow.add_node("clinical_search", clinical_search)
    workflow.add_node("internet_search", internet_search)
    workflow.add_node("final_answer", formulate_final_answer)
    
    # Add edges
    workflow.set_entry_point("process_question")
    
    # Use conditional edges with the router function
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
    
    # Direct edge from final_answer to END
    workflow.add_edge("final_answer", END)
    
    # Compile the graph with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clinical Trials ReAct Agent with LangGraph")
    parser.add_argument("--llm", default="mistral", 
                      choices=["openai", "anthropic", "huggingface", "mistral", "gemini"],
                      help="LLM provider to use")
    parser.add_argument("--embedding", default="huggingface", 
                      choices=["openai", "huggingface", "cohere", "mistral", "biobert", "trial2vec", "BGE-M3"],
                      help="Embedding provider to use")
    parser.add_argument("--model", default="mistral-large-latest", 
                      help="Specific model name (provider-dependent)")
    parser.add_argument("--temperature", type=float, default=0.2, 
                      help="Model temperature (0-1)")
    parser.add_argument("--hf-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                      help="HuggingFace embedding model to use")
    parser.add_argument("--tavily-api-key", 
                      help="Tavily API key for enhanced search")
    return parser.parse_args()

def main():
    """Main function to run the ReAct agent."""
    args = parse_arguments()
    
    # Set environment variables
    if args.hf_embedding_model:
        os.environ["HF_EMBEDDING_MODEL"] = args.hf_embedding_model
    if args.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = args.tavily_api_key
    
    # Create the agent graph
    agent_graph = create_agent_graph(
        llm_provider=args.llm,
        model_name=args.model,
        temperature=args.temperature,
        embedding_provider=args.embedding
    )
    
    if not agent_graph:
        print("Failed to create agent. Exiting.")
        return
    
    print("\n=== Clinical Trials LangGraph ReAct Agent ===")
    print(f"Using LLM: {args.llm}{f' ({args.model})' if args.model else ''}")
    print(f"Using Embeddings: {args.embedding}")
    print("Type 'exit' to quit")
    
    # Initialize chat history
    chat_history = []
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
            
        try:
            # Prepare the input state
            input_state = {
                "messages": [HumanMessage(content=query)],
                "clinical_context": [],
                "internet_context": [],
                "current_task": "start",
                "question": query,
                "all_searches_completed": False
            }
            
            # Run the agent with streaming
            print("\nProcessing your question...")
            
            # Execute the agent graph
            final_state = agent_graph.invoke(
                input_state,
                config={"configurable": {"thread_id": "default_thread"}}
            )
            
            # Extract the final answer
            final_message = final_state["messages"][-1]
            
            # Print the response
            print("\nAnswer:", final_message.content)
            
            # Update chat history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(final_message)
            
            # Keep chat history manageable (last 6 messages - 3 exchanges)
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
