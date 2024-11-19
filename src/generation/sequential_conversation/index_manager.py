import os

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.readers.web import SpiderWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
from loguru import logger

# Initialize ChromaDB client
chroma_db = chromadb.PersistentClient(path="./chroma_db")


def get_pdf_index() -> VectorStoreIndex:
    """Retrieve or create a PDF index using ChromaDB.

    Returns
    -------
    VectorStoreIndex
        The index object that represents the PDF documents stored in the vector database.

    """
    chroma_collection = chroma_db.get_or_create_collection("pdf_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        # Load existing index from ChromaDB
        logger.info("Loading existing PDF index from ChromaDB...")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    else:
        # Create new index if not existing
        logger.info("Creating new PDF index and saving to ChromaDB...")
        parser = LlamaParse(result_type="markdown")
        file_extractor = {".pdf": parser}
        pdf_docs = SimpleDirectoryReader(input_files=["input/business_cases.pdf"], file_extractor=file_extractor).load_data()
        pdf_index = VectorStoreIndex.from_documents(pdf_docs, storage_context=storage_context)
        # Persist the created index in ChromaDB
        pdf_index.storage_context.persist()
        return pdf_index


def get_web_index() -> VectorStoreIndex:
    """Retrieve or create a web index using ChromaDB.

    Returns
    -------
    VectorStoreIndex
        The index object that represents the web documents stored in the vector database.

    """
    chroma_collection = chroma_db.get_or_create_collection("web_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        # Load existing index from ChromaDB
        logger.info("Loading existing Web index from ChromaDB...")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    else:
        # Create new index if not existing
        logger.info("Creating new Web index and saving to ChromaDB...")
        spider_reader = SpiderWebReader(
            api_key=os.environ.get("SPIDER_API_KEY"),
            mode="crawl",
        )
        web_docs = spider_reader.load_data(url="https://www.migrosbank.ch/de/privatpersonen.html")
        web_index = VectorStoreIndex.from_documents(web_docs, storage_context=storage_context)
        # Persist the created index in ChromaDB
        web_index.storage_context.persist()
        return web_index
