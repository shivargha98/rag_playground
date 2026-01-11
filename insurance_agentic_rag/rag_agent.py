from langchain_community.vectorstores import FAISS,Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,BaseMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel,Field

from dotenv import load_dotenv
load_dotenv()


def policy_agent():

    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',) ##embedding model

    vectorstore = Chroma(
    persist_directory="./chromadb", 
    embedding_function=embedding_model,
    collection_name="your_collection_name"  # Optional: defaults to 'langchain'
    )
    # 3. Convert it to a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})