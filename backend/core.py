import os
from typing import Any, Dict, List 
import dotenv
dotenv.load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    docsearch = PineconeVectorStore(embedding= embeddings, index_name = os.environ['INDEX_NAME'])

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain = combine_docs_chain)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    result = qa.invoke(input={"input": query})
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