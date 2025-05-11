import streamlit as st
from chatbot import ask_bot
import pandas as pd
import os

st.title("🧠 Chatbot Trained on Your Data + Q&A History")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        response = ask_bot(query)

    st.success(response)

    # Save to qna_history.csv
    history_file = "data/qna_history.csv"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
    else:
        df = pd.DataFrame(columns=["question", "answer"])

    new_row = pd.DataFrame([{"question": query, "answer": response}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(history_file, index=False)
