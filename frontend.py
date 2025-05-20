# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
import streamlit as st
from langchain_groq import ChatGroq
# Import Deepseek directly in frontend.py
try:
    from langchain_community.llms import DeepseekLLM
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    st.warning("Deepseek integration not available. Using only Groq models.")

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Setup API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

st.set_page_config(page_title="AI Chatbot Agents", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt = st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_DEEPSEEK = ["deepseek-chat", "deepseek-coder"]

# Only show Deepseek option if available
if DEEPSEEK_AVAILABLE:
    provider = st.radio("Select Provider:", ("Groq", "Deepseek"))
else:
    provider = "Groq"

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "Deepseek" and DEEPSEEK_AVAILABLE:
    selected_model = st.selectbox("Select Deepseek Model:", MODEL_NAMES_DEEPSEEK)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

# Define the agent function directly in frontend.py
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider="Groq"):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "Deepseek" and DEEPSEEK_AVAILABLE:
        llm = DeepseekLLM(api_key=DEEPSEEK_API_KEY, model_name=llm_id)
    else:
        # Fallback to Groq if Deepseek is not available
        llm = ChatGroq(model=MODEL_NAMES_GROQ[0])
    
    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

if st.button("Ask Agent!"):
    if user_query.strip():
        with st.spinner("Processing your request..."):
            try:
                response = get_response_from_ai_agent(
                    llm_id=selected_model,
                    query=[user_query],
                    allow_search=allow_web_search,
                    system_prompt=system_prompt,
                    provider=provider
                )
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("If using Deepseek, make sure the Deepseek API package is installed and API key is set.")
    else:
        st.warning("Please enter a query first.")


