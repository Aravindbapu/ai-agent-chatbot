# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()


#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st
from ai_agent import get_response_from_ai_agent

st.set_page_config(page_title="AI Chatbot Agents", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt = st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

if st.button("Ask Agent!"):
    if user_query.strip():
        with st.spinner("Processing your request..."):
            try:
                # Direct call to the agent function instead of API request
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
    else:
        st.warning("Please enter a query first.")


