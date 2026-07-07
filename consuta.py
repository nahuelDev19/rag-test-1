__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]
#%%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#%%
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

chroma_local = Chroma(persist_directory="./vectordb", embedding_function=OpenAIEmbeddings())
#%%
retriever = chroma_local.as_retriever()
#%%
texto = """Eres un asistente amable y cordial que responde preguntas sobre BimBam Buy usando únicamente la información del contexto recuperado.

Reglas:

- Si el usuario saluda, se despide o hace una conversación casual (por ejemplo: "hola", "buen día", "gracias", "¿cómo estás?"), responde de forma amable y natural. No necesitas usar el contexto para estos casos.

- Si el usuario pregunta qué puedes hacer o qué puede consultarte, responde que puedes ayudar a responder preguntas sobre BimBam Buy utilizando la información disponible en los documentos cargados.

- Para cualquier pregunta sobre BimBam Buy, utiliza exclusivamente la información del contexto recuperado.

- Nunca uses conocimiento propio ni información externa al contexto para responder preguntas sobre BimBam Buy.

- Si la respuesta no aparece en el contexto o la pregunta es sobre un tema distinto de BimBam Buy, responde exactamente:
"No tengo esa información en mis documentos."

- No inventes ni completes información que no esté explícitamente en el contexto.

- Responde de forma clara y concisa. Máximo dos oraciones.

"""
#%%


from langchain_core.prompts import ChatPromptTemplate

def prompt(texto):
    system_prompt=(
        texto+
        "\n\n"
        "{context}"
    )
    prompt= ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt
#%%
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


"""def respuesta(pregunta, llm, chroma_db, prompt):
    retriever = chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.65, "k": 4}
    )
    #retriever = chroma_db.as_retriever()

    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, chain)

    results = rag.invoke({"input": pregunta})
    return results
#%%"""
def respuesta(pregunta):
    retriever = chroma_local.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.65, "k": 4}
    )

    chain = create_stuff_documents_chain(llm, prompt(texto))
    rag = create_retrieval_chain(retriever, chain)

    results = rag.invoke({"input": pregunta})
    return results["answer"]

#print(respuesta('cual es la capital de santiago del estero', llm, chroma_local, prompt(texto)))
#resultado = respuesta('cuantos departamentos tiene santiago del estero', llm, chroma_local, prompt(texto))
#print("CONTEXTO USADO:", resultado['context'])
#print("\nRESPUESTA:", resultado['answer'])

#resultados_con_score = chroma_local.similarity_search_with_relevance_scores(
 #   "cuales son los metodos de pago disponibles", k=4
#)
#for doc, score in resultados_con_score:
 #   print(f"Score: {score:.3f} -> {doc.page_content[:60]}")