This folder contains an implementation of chat agents in Langchain.

The file simpleagent.py is a simple OpenAI agent withot RAG.

To use RAG, you must first vectorize the knowledge base. The knowledge_base folder 
contains the code needed to vectorize a set of documents using either 
Pinecone or FAISS. The user needs to set the appropriate values in config.json 
and call vectorizekb.py once.

After this, you can use ragagent.py to ask questions about your knowledge base.

Also note that you need to have your API keys in a .env file in both the
main folder and the knowledge_base folder

