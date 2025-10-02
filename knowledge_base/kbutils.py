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

def ensure_absolute_path(path):
    """
    Convert a relative path to an absolute path, or return the path if already absolute.
    
    Args:
        path (str): The file path to check and potentially convert
    
    Returns:
        str: The absolute path
    """
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(path)

def readcfg(config_file="config.json"):
    """
    Read the configuration file and return it as a dictionary.
    
    Args:
        config_file (str): Path to the config file (default: "config.json")
    
    Returns:
        dict: Configuration dictionary with keys 'embeddings_type' and 'documents_path'
    """
    try:
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

#
# This function loads the API keys but just returns them.
# It is better to set them in the environment variables directly so we don't use this
# function. Instead we use load_and_set_api_keys
#
def load_api_keys():
    """
    Load API keys from .env file.
    
    Returns:
        tuple: (openai_api_key, pinecone_api_key)
    
    Raises:
        ValueError: If required API keys are not found
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # Check if keys are loaded
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    
    return openai_api_key, pinecone_api_key

#
# This function loads the API keys and sets the environment variables,
# which is especially handy for Langchain
#
def load_and_set_api_keys():
    """
    Load API keys from .env file and ensure they're set in os.environ.
    
    Returns:
        dict: Dictionary with the API keys
    
    Raises:
        ValueError: If required API keys are not found
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # Validate
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    
    # Ensure they're in os.environ (in case load_dotenv didn't set them)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    
    return {
        "openai_api_key": openai_api_key,
        "pinecone_api_key": pinecone_api_key
    }

def load_and_chunk_documents(docspath, chunk_size=1000, chunk_overlap=200):
    """
    Load all text and markdown documents from a folder and split them into chunks.
    
    Args:
        docspath (str): Path to the folder containing documents
        chunk_size (int): Size of each text chunk in characters (default: 1000)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 200)
    
    Returns:
        list: List of text chunks (strings)
    """
    # Initialize the text splitter with RecursiveCharacterTextSplitter
    # This is the recommended splitter as it tries to split on natural boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.markdown'}
    
    # Check if the path exists
    if not os.path.exists(docspath):
        raise FileNotFoundError(f"Path '{docspath}' does not exist")
    
    if not os.path.isdir(docspath):
        raise NotADirectoryError(f"Path '{docspath}' is not a directory")
    
    # Iterate through all files in the directory
    for filename in os.listdir(docspath):
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Check if file has supported extension
        if file_ext in supported_extensions:
            filepath = os.path.join(docspath, filename)
            
            try:
                # Read the file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split the content into chunks
                file_chunks = text_splitter.split_text(content)
                chunks.extend(file_chunks)
                
                print(f"Processed {filename}: {len(file_chunks)} chunks")
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return chunks

def vectorize_and_store_pinecone(chunks, index_name="document-embeddings", namespace="default", append=False):
    """
    Vectorize text chunks and store them in Pinecone using LangChain.
    
    Args:
        chunks (list): List of text chunks to vectorize
        index_name (str): Name of the Pinecone index (default: "document-embeddings")
        namespace (str): Pinecone namespace for organizing vectors (default: "default")
        append (bool): If True, add to existing vectors. If False, clear namespace first.
    
    Returns:
        PineconeVectorStore: LangChain vector store object for querying
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change to your preferred region
            )
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")
        # If not appending, clear the namespace first
        if not append:
            print(f"Clearing namespace '{namespace}' before storing new vectors...")
            index = pc.Index(index_name)
            index.delete(delete_all=True, namespace=namespace)
            print(f"Namespace '{namespace}' cleared")
            time.sleep(2)  # Wait for deletion to complete
            print("Deletion complete")
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-ada-002"
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create vector store and add documents
    if append:
        print(f"Appending {len(chunks)} chunks to existing vectors in namespace '{namespace}'...")
    else:
        print(f"Storing {len(chunks)} chunks in namespace '{namespace}'...")

    print(f"Vectorizing and storing {len(chunks)} chunks...")
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    print(f"Successfully stored {len(chunks)} chunks in Pinecone: index %s, namespace %s" % (index_name,namespace))
    
    return vector_store

def load_pinecone_vector_store(index_name="document-embeddings", namespace="default"):
    """
    Load an existing Pinecone vector store.
    
    Args:
        index_name (str): Name of the existing Pinecone index
        namespace (str): Namespace to use (default: "default")
    
    Returns:
        PineconeVectorStore: LangChain vector store object for querying
    """
    # Initialize embeddings (must match what was used to create the vectors)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the existing vector store
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    print(f"Loaded vector store from index '{index_name}' in namespace '{namespace}'")
    
    return vector_store

def vectorize_and_store_faiss(chunks, index_path="./faiss_index", append=False):
    """
    Vectorize text chunks and store them locally using FAISS.
    
    Args:
        chunks (list): List of text chunks to vectorize
        index_path (str): Local path to save the FAISS index
        append (bool): If True, add to existing index. If False, overwrite.
    
    Returns:
        FAISS: LangChain FAISS vector store object
    """
    if not chunks:
        raise ValueError("No chunks provided to vectorize!")
    
    print(f"Vectorizing {len(chunks)} chunks with FAISS...")
    
    # Initialize embeddings (still need OpenAI for embeddings)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Check if we should append to existing index
    if append and os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        # Load existing index
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Add new chunks
        print(f"Adding {len(chunks)} new chunks to existing index...")
        vector_store.add_texts(chunks)

    else:
        # Create new index (overwrites if exists)
        if append and not os.path.exists(index_path):
            print(f"Index doesn't exist at {index_path}, creating new one")
        else:
            print(f"Creating new FAISS index with {len(chunks)} chunks...")
 
        # Create FAISS vector store from texts
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )
    
    # Save to disk
    vector_store.save_local(index_path)
    print(f"FAISS index saved to {index_path}")
    
    return vector_store

def load_faiss_vector_store(index_path="./faiss_index"):
    """
    Load a previously saved FAISS index.
    
    Args:
        index_path (str): Path to the saved FAISS index
    
    Returns:
        FAISS: Loaded vector store
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading pickled data
    )
    
    print(f"FAISS index loaded from {index_path}")
    return vector_store

def load_vector_store(embeddings_type, config={}):
    if embeddings_type == "pinecone":
        pinecone_index = config["pinecone_index"].lower()
        pinecone_namespace = config["pinecone_namespace"].lower()
        return load_pinecone_vector_store(index_name=pinecone_index, namespace=pinecone_namespace)
    elif embeddings_type == "faiss":
        faiss_index_path_tmp = config["faiss_index_path"]
        faiss_index_path = ensure_absolute_path(faiss_index_path_tmp)
        return load_faiss_vector_store(index_path=faiss_index_path)
    else:
        raise Exception("Error in load_vector_store, bad embeddings_type: %s" % embeddings_type)

def search_vector_store(query,vector_store,embeddings_type,kresults,config={}):
    if embeddings_type == "pinecone":
        pinecone_index = config["pinecone_index"].lower()
        pinecone_namespace = config["pinecone_namespace"].lower()
        print(f"Now querying index = {pinecone_index}, namespace={pinecone_namespace}")
        results = vector_store.similarity_search(
            query,
            k=kresults,
            index_name=pinecone_index,
            namespace=pinecone_namespace
        )
        return results
    elif embeddings_type == "faiss":
        faiss_index_path_tmp = config["faiss_index_path"]
        faiss_index_path = ensure_absolute_path(faiss_index_path_tmp)
        results = vector_store.similarity_search(
            query,
            k=kresults
        )
        return results
    else:
        raise Exception("Error in search_vector_store, bad embeddings_type: %s" % embeddings_type)

