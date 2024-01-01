import pandas as pd
import numpy as np
import os
import streamlit as st

import langchain
import openai
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

os.environ['OPENAI_API_KEY'] = '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'

st.title('Chat with the Database 💬')
file = st.file_uploader("Upload a CSV file", type=["csv"])

data = pd.DataFrame(columns=['Date', 'Time', 'Query', 'Answer'])

if file is not None:
    col1, col2 = st.columns(2)
    with st.expander("Show Database"):  
        df = pd.read_csv(file, encoding = "ISO-8859-1")
        st.dataframe(df.head())
        data = df.copy()
    
    with st.expander("Describe Database"):
        st.dataframe(data.describe())
        
    query = st.text_input("Enter your query here")
    
    if query:
        st.write("Your query is: ", query)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)
        answer = agent.run(query)  
        st.write("Answer:", answer)
