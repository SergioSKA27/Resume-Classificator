import streamlit as st
import pandas as pd  
import requests
import torch
from datasets import load_dataset
from sentence_transformers.util import semantic_search
import time


st.set_page_config(layout='wide')
model_id = "sentence-transformers/all-MiniLM-L6-v2"

st.title('Calificador de CV')
st.caption('Calificador de cv  basado en similaridad semantica.')
st.divider()

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

@st.cache_data
def load_data():
    dataframe = load_dataset('e-sdmartinez/Resumes')
    return dataframe

dataframe = load_data()
#dataframe = pd.read_csv('Resume.csv')
dataset_embeddings = return_embedings()

st.header('Original Data')
st.dataframe(pd.DataFrame(dataframe['train']))

st.header('Embeddings')

st.dataframe(dataset_embeddings)



prompt = st.text_area('Ingrese una descripcion del candidato', placeholder="PROGRAM MANAGER/BUSINESS ANALYST,AMC COMPUTER SPECIALIST AND INTERN,DATABASE PROGRAMMER/ANALYST (.NET DEVELOPER)")
re = prompt.upper() if len(prompt) > 0 else  "PROGRAM MANAGER/BUSINESS ANALYST,AMC COMPUTER SPECIALIST AND INTERN,DATABASE PROGRAMMER/ANALYST (.NET DEVELOPER)"

query_embeddings = torch.FloatTensor(embedd_prompt(re))

st.header('Embedd Prompt')

with st.expander('Ver embedding'):
    st.write(query_embeddings)


top = st.slider('Seleccione el numero de perfiles a seleccionar',min_value=5, max_value=30)
_,bcol = st.columns([.7,.3])

if bcol.button('🔍Buscar',use_container_width=True):
    with st.spinner('Buscando Candidatos...'):
        time.sleep(2)
        hits = semantic_search(query_embeddings, dataset_embeddings, top_k=top)
    st.toast('Candidatos Encontrados',icon='🚀')

    st.header('Resultados')

    for i in range(len(hits[0])):
        st.write(':blue[Score:]', hits[0][i]['score'])
        try:
            with st.expander('CV HTML'):
                st.write(dataframe[hits[0][i]['corpus_id']]['Resume_html'],unsafe_allow_html=True)
            with st.expander('CV STR'):
                st.write(dataframe.iloc[hits[0][i]['corpus_id']]['Resume_str'])
        except:
            with st.expander('HTML'):
                st.write(dataframe['train'][hits[0][i]['corpus_id']]['Resume_html'],unsafe_allow_html=True)
            with st.expander('TEXT'):
                st.write(dataframe['train'][hits[0][i]['corpus_id']]['Resume_str'])


        st.divider()
    st.write(hits)
