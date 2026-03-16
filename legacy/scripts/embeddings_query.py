import os

from cicada.core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

model_name = os.getenv("MODEL", "bge-m3")

embedding_model = Embeddings(
    api_key=API_KEY,
    api_base_url=BASE_URL,
    model_name=model_name,
)

# Generate embeddings for a list of texts
texts = ["This is a test document.", "Another test document."]
embeddings = embedding_model.embed_documents(texts)
print(embeddings)

# Generate an embedding for a single query
query_embedding = embedding_model.embed_query("This is a query.")
print(query_embedding)
