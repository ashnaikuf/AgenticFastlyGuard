from qdrant_client import QdrantClient, models

from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

load_dotenv()
# Access the variables securely
api_key = os.getenv("QDRANT_APIKEY")
database_url = os.getenv("CLUSTER_ENDPOINT")

print(f"Loaded API Key: {len(api_key)} and DB URL: {len(database_url)}")

client = QdrantClient(
    database_url,
    api_key=api_key,
    cloud_inference=True,
    timeout=30,
)

collection_name="fastly_wafrules"

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense_vector": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    },
    sparse_vectors_config={
        "bm25_sparse_vector": models.SparseVectorParams(
            modifier=models.Modifier.IDF # Enable Inverse Document Frequency
        )
    }
)