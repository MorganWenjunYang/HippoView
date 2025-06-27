#!/usr/bin/env python3
"""
Streamlit Frontend for HippoView Clinical Trials Agent

This app provides a user-friendly web interface for the Clinical Trials ReAct Agent
with RAG capabilities and internet search functionality.
"""

import os
import sys
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import traceback

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules

from agent.ReAct_LG_with_MCP import create_agent_graph, check_mcp_server_status
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hippoview-streamlit")

# Page configuration
st.set_page_config(
    page_title="HippoView - Clinical Trials AI Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .error-message {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .info-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent_executor' not in st.session_state:
        st.session_state.agent_executor = None
    
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = "ReAct LG with MCP"
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if 'mcp_server_status' not in st.session_state:
        st.session_state.mcp_server_status = None

def create_sidebar():
    """Create the sidebar with configuration options."""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Agent Type Selection
        agent_type = st.selectbox(
            "Agent Type",
            ["ReAct LG with MCP"],
            help="Choose between LangChain ReAct agent or LangGraph agent with MCP server"
        )
        
        # LLM Configuration
        st.subheader("🧠 LLM Settings")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["deepseek", "mistral", "openai", "anthropic", "huggingface", "gemini"],
            help="Choose the language model provider"
        )
        
        model_name = st.text_input(
            "Model Name (Optional)",
            placeholder="e.g., deepseek-chat, mistral-large-latest",
            help="Specific model name for the provider"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Controls randomness in responses (0=deterministic, 1=creative)"
        )
        
        # Embedding Configuration
        st.subheader("🔍 Embedding Settings")
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["huggingface", "openai", "cohere", "mistral", "deepseek"],
            help="Choose the embedding model provider"
        )
        
        # MCP Server Configuration (only for MCP agent)
        if agent_type == "ReAct LG with MCP":
            st.subheader("🔗 MCP Server")
            mcp_server_url = st.text_input(
                "MCP Server URL",
                value="http://127.0.0.1:8000/mcp",
                help="URL of the MCP server for RAG functionality"
            )
            
            # Check MCP server status
            if st.button("Check MCP Server Status"):
                with st.spinner("Checking MCP server..."):
                    try:
                        # Use asyncio to run the async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        status = loop.run_until_complete(check_mcp_server_status(mcp_server_url))
                        st.session_state.mcp_server_status = status
                        loop.close()
                        
                        if status:
                            st.success("✅ MCP Server is running")
                        else:
                            st.error("❌ MCP Server is not accessible")
                    except Exception as e:
                        st.error(f"❌ Error checking MCP server: {str(e)}")
                        st.session_state.mcp_server_status = False
        
        # Initialize Agent Button
        st.subheader("🚀 Agent Control")
        if st.button("Initialize Agent", type="primary"):
            with st.spinner("Initializing agent..."):
                try:
                    # if agent_type == "ReAct LC":
                    #     # Initialize LangChain ReAct agent
                    #     agent_executor = create_react_agent_with_rag(
                    #         llm_provider=llm_provider,
                    #         model_name=model_name if model_name else None,
                    #         temperature=temperature,
                    #         embedding_provider=embedding_provider
                    #     )
                        
                    #     if agent_executor:
                    #         st.session_state.agent_executor = agent_executor
                    #         st.session_state.agent_type = agent_type
                    #         st.session_state.agent_initialized = True
                    #         st.success("✅ ReAct LC Agent initialized successfully!")
                    #     else:
                    #         st.error("❌ Failed to initialize ReAct LC Agent")
                    #         st.session_state.agent_initialized = False
                    
                    if agent_type == "ReAct LG with MCP":
                        # Initialize LangGraph agent with MCP
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        agent_graph = loop.run_until_complete(create_agent_graph(
                            llm_provider=llm_provider,
                            model_name=model_name if model_name else None,
                            temperature=temperature,
                            mcp_server_url=mcp_server_url
                        ))
                        
                        loop.close()
                        
                        if agent_graph:
                            st.session_state.agent_executor = agent_graph
                            st.session_state.agent_type = agent_type
                            st.session_state.agent_initialized = True
                            st.success("✅ ReAct LG with MCP Agent initialized successfully!")
                        else:
                            st.error("❌ Failed to initialize ReAct LG with MCP Agent")
                            st.session_state.agent_initialized = False
                    
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
                    st.session_state.agent_initialized = False
                    logger.error(f"Agent initialization error: {e}")
                    st.code(traceback.format_exc())
        
        # Agent Status
        if st.session_state.agent_initialized:
            st.success(f"✅ {st.session_state.agent_type} Agent Ready")
        else:
            st.warning("⚠️ Agent not initialized")
        
        # Clear Chat Button
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

def display_chat_messages():
    """Display the chat messages in the main area."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', 
                          unsafe_allow_html=True)
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', 
                          unsafe_allow_html=True)
        elif message["role"] == "error":
            with st.chat_message("assistant"):
                st.markdown(f'<div class="chat-message error-message">❌ {message["content"]}</div>', 
                          unsafe_allow_html=True)

def process_user_query(query: str) -> str:
    """Process user query with the initialized agent."""
    if not st.session_state.agent_initialized or not st.session_state.agent_executor:
        return "❌ Agent not initialized. Please initialize the agent first."
    
    try:
        # if st.session_state.agent_type == "ReAct LC":
        #     # LangChain ReAct agent
        #     chat_history = []
        #     for msg in st.session_state.messages[-6:]:  # Last 6 messages for context
        #         if msg["role"] == "user":
        #             chat_history.append(HumanMessage(content=msg["content"]))
        #         elif msg["role"] == "assistant":
        #             chat_history.append(AIMessage(content=msg["content"]))
            
        #     response = st.session_state.agent_executor.invoke({
        #         "input": query,
        #         "chat_history": chat_history
        #     })
            
        #     return response.get("output", "No response generated")
        
        if st.session_state.agent_type == "ReAct LG with MCP":
            # LangGraph agent with MCP
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "clinical_context": [],
                "internet_context": [],
                "current_task": "process_question",
                "question": query,
                "all_searches_completed": False
            }
            
            # Run the agent WITH REQUIRED CONFIG
            final_state = loop.run_until_complete(
                st.session_state.agent_executor.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": "default_thread"}}
                )
            )
            loop.close()
            
            # Extract the final answer
            if final_state and "messages" in final_state:
                for message in reversed(final_state["messages"]):
                    if hasattr(message, 'content') and message.content:
                        return message.content
            
            return "No response generated from LangGraph agent"
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<div class="main-header">🔬 Clinical Trials AI Assistant</div>', 
                unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with Clinical Trials Expert")
        
        # Display welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
            <div class="info-box">
                <h4>👋 Welcome to HippoView!</h4>
                <p>I'm your Clinical Trials AI Assistant. I can help you with:</p>
                <ul>
                    <li>🔍 Finding specific clinical trials</li>
                    <li>📋 Understanding eligibility criteria</li>
                    <li>💊 Learning about treatments being studied</li>
                    <li>🌐 Searching for current medical research</li>
                    <li>📊 Analyzing clinical trial data</li>
                </ul>
                <p><strong>To get started:</strong> Initialize the agent using the sidebar, then ask me anything about clinical trials!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        display_chat_messages()
        
        # Chat input
        if prompt := st.chat_input("Ask me about clinical trials..."):
            if not st.session_state.agent_initialized:
                st.error("❌ Please initialize the agent first using the sidebar.")
            else:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Process query and get response
                with st.spinner("Thinking..."):
                    response = process_user_query(prompt)
                
                if response.startswith("❌"):
                    # Error response
                    st.session_state.messages.append({"role": "error", "content": response})
                else:
                    # Normal response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Refresh the page to show the updated conversation with input box positioned correctly
                st.rerun()
    
    with col2:
        st.subheader("📊 System Status")
        
        # Agent status
        if st.session_state.agent_initialized:
            st.success(f"✅ {st.session_state.agent_type}")
        else:
            st.warning("⚠️ No agent initialized")
        
        # MCP server status (if applicable)
        if st.session_state.agent_type == "ReAct LG with MCP":
            if st.session_state.mcp_server_status is True:
                st.success("✅ MCP Server Connected")
            elif st.session_state.mcp_server_status is False:
                st.error("❌ MCP Server Disconnected")
            else:
                st.info("ℹ️ MCP Server Status Unknown")
        
        # Chat statistics
        if st.session_state.messages:
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            error_msgs = len([m for m in st.session_state.messages if m["role"] == "error"])
            
            st.metric("User Messages", user_msgs)
            st.metric("Assistant Responses", assistant_msgs)
            if error_msgs > 0:
                st.metric("Errors", error_msgs)
        
        # Recent activity
        if st.session_state.messages:
            st.subheader("📝 Recent Activity")
            recent_messages = st.session_state.messages[-3:]
            for msg in recent_messages:
                role_emoji = {"user": "👤", "assistant": "🤖", "error": "❌"}
                timestamp = datetime.now().strftime("%H:%M")
                st.text(f"{role_emoji.get(msg['role'], '💬')} {timestamp}")
        
        # Help section
        st.subheader("❓ Need Help?")
        with st.expander("Example Questions"):
            st.markdown("""
            - "Find diabetes treatment trials"
            - "What are the eligibility criteria for cancer studies?"
            - "Show me recent COVID-19 vaccine trials"
            - "What clinical trials are available for heart disease?"
            - "Explain the phases of clinical trials"
            """)
        
        with st.expander("Troubleshooting"):
            st.markdown("""
            **Agent not responding?**
            - Check if the agent is initialized
            - Verify your API keys are set
            - Try reinitializing the agent
            
            **MCP Server issues?**
            - Ensure the server is running
            - Check the server URL
            - Verify network connectivity
            """)

if __name__ == "__main__":
    main()
