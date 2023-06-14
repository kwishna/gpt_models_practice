"""
This program reads the user data and uses openai ChatGPT.
Based on your data, it returns the response against your Query.
"""

import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

os.environ["OPENAI_API_KEY"] = 'sk-??'

documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# storage_context = StorageContext.from_defaults(persist_dir='./data')
# index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
res = query_engine.query("What was the name of the person living on Mars?")

print(res)
