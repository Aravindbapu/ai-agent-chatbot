# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Setup API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

st.set_page_config(page_title="AI Chatbot Agents", layout="centered")
st.title("AI Chatbot Agent")
st.write("Create and Interact with the AI Agents!")

system_prompt = st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

# Simplified to only use Groq for now
selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

# Define the agent function directly in frontend.py
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id)
    
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
                    system_prompt=system_prompt
                )
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query first.")

