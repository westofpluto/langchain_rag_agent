###########################################################
# This file contains code to create a vector store
# for use in a RAG agent
##########################################################
import json
import sys
import os 
import time

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter

from kbutils import (
    ensure_absolute_path,
    readcfg,
    load_and_set_api_keys,
    load_and_chunk_documents,
    vectorize_and_store_pinecone,
    load_pinecone_vector_store,  
    vectorize_and_store_faiss,
    load_faiss_vector_store
)

if __name__ == "__main__":

    #
    # start by getting keys. This function actually sets the keys needed as environment
    # variables, so we don't need the returned keys dictionary
    #
    load_and_set_api_keys()

    cfg = readcfg()
    path = cfg["documents_path"]
    abs_docs_path = ensure_absolute_path(path)
    embeddings_type = cfg["embeddings_type"].lower()
    embeddings_action = cfg["embeddings_action"].lower()
    pinecone_index = cfg["pinecone_index"].lower()
    pinecone_namespace = cfg["pinecone_namespace"].lower()
    faiss_index_path_tmp = cfg["faiss_index_path"]
    faiss_index_path = ensure_absolute_path(faiss_index_path_tmp)

    print(json.dumps(cfg,indent=4)) 
 
    print("Reading documents from path %s" % abs_docs_path)
    print("Using embeddings vectors in %s" % embeddings_type)
    if embeddings_type == "pinecone":
        if embeddings_action == "create":
            print("Creating and storing Pinecone embedding vectors in index %s" % pinecone_index)
        elif embeddings_action == "append":
            print("Appending new embeddings Pinecone vectors to existing index %s" % pinecone_index)
        elif embeddings_action == "load":
            print("Loading Pinecone embeddings vectors for index=%s" % pinecone_index)
    elif embeddings_type == "faiss":
        if embeddings_action == "create":
            print("Creating and storing FAISS embedding vectors in path %s" % faiss_index_path)
        elif embeddings_action == "append":
            print("Appending new embeddings FAISS vectors to existing path %s" % faiss_index_path)
        elif embeddings_action == "load":
            print("Loading FAISS embeddings vectors for path=%s" % faiss_index_path)
    else:
        raise Exception("Bad embeddings_type: %s" % embeddings_type)

    chunks = load_and_chunk_documents(abs_docs_path, chunk_size=450, chunk_overlap=100)
    for i,chunk in enumerate(chunks):
        print("CHUNK %d" % i)
        print(chunk)
        print("")
        print("***************************************************")
        print("")

    append=False
    if embeddings_type == "pinecone":
        print("Testing Pinecone")
        if embeddings_action == "create": 
            print("Creating Pinecone index %s, pinecone namespace %s" % (pinecone_index,pinecone_namespace))
            # Then vectorize and store in Pinecone
            vector_store = vectorize_and_store_pinecone(
                chunks=chunks,
                index_name=pinecone_index,
                namespace=pinecone_namespace
            )
        elif embeddings_action == "append": 
            print("Appending vectors to Pinecone index %s, Pinecone namespace %s" % (pinecone_index,pinecone_namespace))
            # Then vectorize and store in Pinecone
            vector_store = vectorize_and_store_pinecone(
                chunks=chunks,
                index_name=pinecone_index,
                namespace=pinecone_namespace,
                append=True
            )
        elif embeddings_action == "load":
            print("Loading Pinecone index %s, namespace %s" % (pinecone_index,pinecone_namespace))
            vector_store = load_pinecone_vector_store(
                index_name=pinecone_index,    
                namespace=pinecone_namespace
            )

        else:
            raise Exception("Bad embeddings action: %s" % embeddings_action)

        # Now you can query the vector store
        print("Waiting for vector store to complete")
        time.sleep(10)
        print(f"Now querying index = {pinecone_index}, namespace={pinecone_namespace}")
        results = vector_store.similarity_search(
            "What is the recipe for apple pie?", 
            k=3, 
            index_name=pinecone_index,
            namespace=pinecone_namespace
        )
        print("RESULTS:")
        print("Found %d results" % len(results))
        for doc in results:
            print("DOC: ")
            print(doc.page_content)
            print("******************************")

    elif embeddings_type == "faiss":
        print("Testing FAISS")
        if embeddings_action == "create":
            print("Creating FAISS vector store at path %s" % faiss_index_path)
            # Then vectorize and store in FAISS    
            vector_store = vectorize_and_store_faiss(chunks, index_path=faiss_index_path)

        elif embeddings_action == "append":
            print("Appending vectors to FAISS vector store at path %s" % faiss_index_path)
            # Then vectorize and store in FAISS    
            vector_store = vectorize_and_store_faiss(chunks, index_path=faiss_index_path, append=True)

        elif embeddings_action == "load":
            print("Loading FAISS vector store at index path %s" % faiss_index_path)
            vector_store = load_faiss_vector_store(index_path=faiss_index_path)

        else:
            raise Exception("Bad embeddings action: %s" % embeddings_action)

        # Now you can query the vector store
        print("Waiting for vector store to complete")
        time.sleep(10)
        print(f"Now querying FAISS vector store at path {faiss_index_path}")
        results = vector_store.similarity_search(
            "What is the recipe for apple pie?", 
            k=3
        )
        print("RESULTS:")
        print("Found %d results" % len(results))
        for doc in results:
            print("DOC: ")
            print(doc.page_content)
            print("******************************")

    print("")
    print("")
    print("Done!")
    sys.exit(0)

