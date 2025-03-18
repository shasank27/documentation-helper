import os
from typing import Any, Dict, List 
import dotenv
dotenv.load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    docsearch = PineconeVectorStore(embedding= embeddings, index_name = os.environ['INDEX_NAME'])

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm = llm, retriever=docsearch.as_retriever(), prompt= rephrase_prompt
    )

    qa = create_retrieval_chain(history_aware_retriever, combine_docs_chain = combine_docs_chain)
    
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result


if __name__ == "__main__":
    query = "What is a Langchain Chain?"
    res = run_llm(query)
    print(res["result"])