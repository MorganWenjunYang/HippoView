# ReAct.py

# 1. define the ReAct agent
# 2. define the multi-agent structure
# 3. define the task for the agent

# enable agent to access internet 
# enable agent to use tools
# enable agent to run code

import os
import sys
from typing import List, Dict, Any, Optional
import argparse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from RAG directory
from RAG.rag import get_llm_model, get_embedding_model, setup_retriever, fetch_trials_from_mongo, transform_trials_to_documents, create_vectorstore

def create_search_tools() -> List[Tool]:
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
                name="DuckDuckGo Search",
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
                name="Wikipedia",
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
                name="ArXiv",
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
                    name="Tavily Search",
                    func=tavily_search.run,
                    description="A more powerful search engine that provides structured, high-quality information. Use this for complex queries requiring accurate and up-to-date information."
                )
            )
            print("Added Tavily search tool")
        except ImportError:
            print("Could not import Tavily search. Install with: pip install tavily-python")
    
    return tools

def create_react_agent_with_rag(llm_provider="mistral", model_name=None, temperature=0.2, embedding_provider="huggingface"):
    """Create a ReAct agent with RAG capabilities and internet search.
    
    Args:
        llm_provider: The LLM provider to use
        model_name: Specific model name
        temperature: Temperature for generation
        embedding_provider: Provider for embeddings
        
    Returns:
        An AgentExecutor with ReAct agent and tools
    """
    # Get LLM
    llm = get_llm_model(provider=llm_provider, model_name=model_name, temperature=temperature)
    
    # Get or create vectorstore for RAG
    print("Setting up clinical trials knowledge base...")
    trials = fetch_trials_from_mongo()
    if not trials:
        print("No trials found in MongoDB. Please check your database connection.")
        return None
    
    documents = transform_trials_to_documents(trials)
    vectorstore = create_vectorstore(documents, embedding_provider=embedding_provider)
    if not vectorstore:
        print("Failed to create vector store. Please check your embedding configuration.")
        return None
    
    # Set up retriever
    retriever = setup_retriever(vectorstore)
    
    # Create retriever tool for clinical trials database
    retriever_tool = create_retriever_tool(
        retriever,
        name="ClinicalTrialsDB",
        description="Search for information about clinical trials in the database. Use this for questions about specific clinical trials, medical treatments being studied, or patient eligibility criteria."
    )
    
    # Get internet search tools
    search_tools = create_search_tools()
    
    # Combine all tools
    all_tools = [retriever_tool] + search_tools
    
    # Create the complete system prompt with ReAct format
    system_prompt = """You are a clinical trials expert assistant that can search both a clinical trials database and the internet.

To answer questions effectively:
1. For questions about specific trials, trial details, eligibility criteria, or treatments being studied, use the ClinicalTrialsDB tool first.
2. For general medical information, current research, or context not in the database, use the internet search tools.
3. You can combine information from multiple sources to provide comprehensive answers.

Always think step by step:
1. Consider which tool is most appropriate for the question
2. Use the tool to gather information
3. If needed, use additional tools to supplement your knowledge
4. Synthesize the information and provide a clear, accurate response

Always cite your sources clearly, indicating which information came from the clinical trials database versus internet sources.
Keep responses concise, factual and evidence-based.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
    
    # Modify your prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ]).partial(tools=", ".join([t.name for t in all_tools]), tool_names=", ".join([t.name for t in all_tools]))

    # Create the ReAct agent with proper formatting
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
        )
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=None,
        early_stopping_method="generate"
    )
    
    return agent_executor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clinical Trials ReAct Agent with RAG and Internet Search")
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
    
    # Create the agent
    agent_executor = create_react_agent_with_rag(
        llm_provider=args.llm,
        model_name=args.model,
        temperature=args.temperature,
        embedding_provider=args.embedding
    )
    
    if not agent_executor:
        print("Failed to create agent. Exiting.")
        return
    
    print("\n=== Clinical Trials ReAct Agent ===")
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
            # Run the agent
            response = agent_executor.invoke({
                "input": query,
                "chat_history": chat_history
            })
            
            # Print the response
            print("\nAnswer:", response["output"])
            
            # Update chat history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=response["output"]))
            
            # Keep chat history manageable (last 6 messages - 3 exchanges)
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

