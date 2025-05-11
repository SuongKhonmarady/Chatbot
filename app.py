import streamlit as st
from chatbot import ask_bot

st.title("📚 Local Custom Chatbot")

query = st.text_input("Ask a question based on your data:")
if query:
    with st.spinner("Thinking..."):
        response = ask_bot(query)
    st.success(response)
    