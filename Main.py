import streamlit as st
import pandas as pd  
import requests
import torch
from datasets import load_dataset
from sentence_transformers.util import semantic_search


model_id = "sentence-transformers/all-MiniLM-L6-v2"



api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


@st.cache_data
def return_embedings():
    faqs_embeddings = load_dataset('e-sdmartinez/cvembeddings')
    dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)
    return dataset_embeddings

@st.cache_data
def embedd_prompt(prompt):
    question = [prompt]
    output = query(question)
    return output



dataframe = pd.read_csv('Resume.csv')
dataset_embeddings = return_embedings()

st.header('Original Data')
st.dataframe(dataframe)

st.header('Embeddings')

st.dataframe(dataset_embeddings)



prompt = st.text_area('Ingrese una descripcion del candidato', placeholder="PROGRAM MANAGER/BUSINESS ANALYST,AMC COMPUTER SPECIALIST AND INTERN,DATABASE PROGRAMMER/ANALYST (.NET DEVELOPER)")
re = prompt.upper() if len(prompt) > 0 else  "PROGRAM MANAGER/BUSINESS ANALYST,AMC COMPUTER SPECIALIST AND INTERN,DATABASE PROGRAMMER/ANALYST (.NET DEVELOPER)"

query_embeddings = torch.FloatTensor(embedd_prompt(re))

st.header('Embedd Prompt')

with st.expander('Ver embedding'):
    st.write(query_embeddings)


top = st.slider('Seleccione el numero de perfiles a seleccionar',min_value=5, max_value=30)
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=top)

st.header('Resultados')
st.write(hits)

for i in range(len(hits[0])): 
    with st.expander('CV HTML'):
        st.write(dataframe.iloc[hits[0][i]['corpus_id']]['Resume_html'],unsafe_allow_html=True)
    with st.expander('CV STR'):
        st.write(dataframe.iloc[hits[0][i]['corpus_id']]['Resume_str'])
    st.divider()
