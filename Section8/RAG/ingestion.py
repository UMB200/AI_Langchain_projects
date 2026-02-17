import asyncio
import os
import ssl
import certifi
from typing import Any, Dict, List
from chromadb import Metadata
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
from dotenv import load_dotenv

load_dotenv()

# SSL context configuration in order to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

# Chroma
vector_store  = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
#vector_store = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int=50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)

    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/ {len(batches)} ({len(batch)} documents)")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
     # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful_batches = sum(1 for result in results if results is True)

    if successful_batches == len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully ({successful_batches}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful_batches}/ {len(batches)} batches successfully")

async def main():
    """Main async func to control entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    url_path = "https://python.langchain.com"
   
    log_info("🔍 TavilyMap: Starting to crawl documentation from https://python.langchain.com/", Colors.PURPLE)

    # Crawl doc site
    tavily_crawl_result = tavily_crawl.invoke({
        "url" : url_path,
        "max_depth": 2,
        "extract_depth": "advanced",
        #"instructions": "content on ai agents"
    })
    # full_docs_list = [{Document(
    #     page_content=result['results'], 
    #     metadata={"source": result['url']}) for result in tavily_crawl_result["results"]}]

    # Convert Tavily crawl results to LangChain Document objects
    full_docs_list = []
    for tavily_crawl_result_item in tavily_crawl_result['results']:
        log_info(f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from doc site")
        full_docs_list.append(
            Document(page_content=tavily_crawl_result_item["raw_content"], 
            metadata={"source": tavily_crawl_result_item["url"]})
        )
    
    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Splitter: Processing {len(full_docs_list)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(full_docs_list)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(full_docs_list)} documents")

    #Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline finished succcessfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f" -> Documents extracted: {len(full_docs_list)}")
    log_info(f" -> Chunks created {len(splitted_docs)}")

    
def chunk_urls(urls: List[str], chunk_size: int=20)-> List[List[str]]:
    """Split URLs into chunks"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num:int) -> List[Dict[str, Any]]:
    """Extract doc from batch of URL"""
    try:
        log_info(f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.invoke(input={"urls": urls})
        log_success(f"TavilyExtract Completed batch {batch_num} - extracted {len(docs.get('results', []))} dcouments")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

if __name__ == "__main__":
    asyncio.run(main())