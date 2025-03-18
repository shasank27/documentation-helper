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

def ingest_docs2():
    from langchain_community.document_loaders import FireCrawlLoader
    langchain_docs_base_url = [
        "https://python.langchain.com/docs/concepts/chat_models/",
        "https://python.langchain.com/docs/concepts/messages/",
        "https://python.langchain.com/docs/concepts/prompt_templates/",
        "https://python.langchain.com/docs/concepts/example_selectors/",
        "https://python.langchain.com/docs/concepts/output_parsers/",
        "https://python.langchain.com/docs/concepts/document_loaders/",
        "https://python.langchain.com/docs/concepts/text_splitters/",
        "https://python.langchain.com/docs/concepts/embedding_models/",
        "https://python.langchain.com/docs/concepts/vector_stores/",
        "https://python.langchain.com/docs/concepts/retrievers/",
        "https://python.langchain.com/docs/concepts/tools/",
        "https://python.langchain.com/docs/concepts/agents/",
        "https://python.langchain.com/docs/concepts/callbacks/"
    ]
    langchain_docs_base_url2 = langchain_docs_base_url[0:1]
    for url in langchain_docs_base_url2:
        print(f"Firecrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "crawlerOptions":{"limit": 5},
                "pageOptions": {"onlyMainContent": True},
                "wait_until_done": True,
            }
        )

        docs = loader.load()
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print("*** Done ***")

if __name__ == "__main__":
    ingest_docs2()

    # llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash")
    
    # query = "What is Vector?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)
    
    # vectorstore = PineconeVectorStore(embedding= embeddings, index_name = os.environ['INDEX_NAME'])