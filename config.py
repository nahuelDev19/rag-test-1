
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

persistent_client = chromadb.PersistentClient(path='./vectordb')
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(client=persistent_client, collection_name="langchain", embedding_function=embeddings)

def add_files_to_vectordb(filepath):
    existing_sources = get_unique_sources_list()
    if filepath.split('/')[-1] in existing_sources:
        print(f"Ya indexado, salteando: {filepath}")
        return

    loader = PyPDFLoader(filepath)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)

def get_unique_sources_list():
    collection_data = persistent_client.get_collection('langchain').get(include=['metadatas'])
    metadatas = collection_data['metadatas']
    sources = {m.get('source') for m in metadatas if m.get('source')}
    return list({s.split('/')[-1] for s in sources})

add_files_to_vectordb('./documentacion/Guía de Tiempos y Costos de Envío de BimBam Buy.pdf')
add_files_to_vectordb('./documentacion/Programa de Afiliados de BimBam Buy.pdf')
add_files_to_vectordb('./documentacion/Preguntas Frecuentes sobre Métodos de Pago de.pdf')
add_files_to_vectordb('./documentacion/Política de Reembolsos y Devoluciones de BimBam.pdf')
add_files_to_vectordb('./documentacion/Manual de Garantía de Productos de BimBam Buy.pdf')