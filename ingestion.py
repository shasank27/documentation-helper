import os 
import dotenv
dotenv.load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going top add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name = "langchain-doc-index")
    print("*** Loading done ***")

if __name__ == "__main__":
    ingest_docs()

    # llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash")
    
    # query = "What is Vector?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)
    
    # vectorstore = PineconeVectorStore(embedding= embeddings, index_name = os.environ['INDEX_NAME'])