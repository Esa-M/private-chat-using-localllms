from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
import sys
#**Step 1: Load the PDF File from Data Path****

loader = PyPDFLoader("your document path here")
documents=loader.load()


text_splitter=RecursiveCharacterTextSplitter(
                                             chunk_size=800,
                                             chunk_overlap=80)


text_chunks=text_splitter.split_documents(documents)



embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})



vector_store=FAISS.from_documents(text_chunks, embeddings)



llm=CTransformers(model="guff / ggml model path here",
                  model_type="llama",
                  config={'max_new_tokens':2048,
                          'context_length':4096,
                          'temperature':0.01})


template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])


chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})



response=chain({'query':"your query here"})

print(response["result"])

