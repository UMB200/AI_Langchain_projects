import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    print("RAG Ingesting file")
    file_path = os.path.join(os.path.dirname(__file__), "mediumblog.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print("Started splitting..........")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("Start ingesting chunks into Pinecone..........")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=os.getenv("INDEX_NAME")
    )
    print("Ingestion complete..........")