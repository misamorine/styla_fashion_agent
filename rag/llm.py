import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL = "deepseek/deepseek-chat-v3-0324"


def get_stylist_llm(temperature: float = 0.4, max_tokens: int = 800):
    """Returns a configured LangChain ChatOpenAI instance supporting OpenRouter or OpenAI."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    is_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    model_name = MODEL if is_openrouter else "gpt-3.5-turbo"
    base_url = "https://openrouter.ai/api/v1" if is_openrouter else None

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate(prompt: str) -> str:
    """Generates LLM response using LangChain ChatOpenAI model, with graceful fallback."""
    llm = get_stylist_llm()
    if not llm:
        return "### 💡 Expert Styling Advice (Offline / Rule-Based Mode)\n\n" + prompt.replace("You are an expert AI Fashion Stylist.", "").strip()

    try:
        messages = [
            SystemMessage(content="You are an expert fashion stylist."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"### 💡 Expert Styling Advice (Rule-Based Fallback)\n\n{prompt}\n\n*(Note: LLM API returned: {e})*"
