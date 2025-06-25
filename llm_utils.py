import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

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
    
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return ChatDeepSeek(model_name=model_name or "deepseek-chat", temperature=temperature)
    
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