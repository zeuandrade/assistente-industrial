
import streamlit as st
import pandas as pd
import os
import openai
from langchain_community.llms import HuggingFaceEndpoint
from langchain.agents import create_pandas_dataframe_agent

# Interface inicial
st.set_page_config(page_title="Assistente Industrial", layout="wide")
st.title("🤖 Assistente de Manutenção Industrial")
st.markdown("Faça perguntas sobre sua base de ordens de serviço. Ex: 'Qual equipamento teve mais falhas?'")

# Upload do CSV
csv_file = st.file_uploader("📤 Envie sua planilha CSV de manutenção", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file, sep=";")
    st.success("CSV carregado com sucesso!")
    st.dataframe(df.head())

    # Configurar modelo LLM com HuggingFace
    huggingface_api_key = st.secrets["HUGGINGFACE_API_KEY"]
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=huggingface_api_key
    )

    # Criar agente para perguntas
    agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=False)

    # Caixa de perguntas
    pergunta = st.text_input("💬 Sua pergunta sobre os dados:")
    if pergunta:
        with st.spinner("Pensando..."):
            resposta = agent.run(pergunta)
        st.success("✅ Resposta:")
        st.write(resposta)
