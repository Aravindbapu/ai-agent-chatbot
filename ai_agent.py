# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup API Keys for Groq, Deepseek and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
DEEPSEEK_API_KEY=os.environ.get("DEEPSEEK_API_KEY")  # Changed from OPENAI to DEEPSEEK

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_community.llms import DeepseekLLM  # Import Deepseek instead of OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

deepseek_llm=DeepseekLLM(api_key=DEEPSEEK_API_KEY, model_name="deepseek-chat")  # Changed from OpenAI
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="Deepseek":  # Changed from OpenAI to Deepseek
        llm=DeepseekLLM(api_key=DEEPSEEK_API_KEY, model_name=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]


