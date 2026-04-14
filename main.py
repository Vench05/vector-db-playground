import shutil
from pprint import pprint

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = None
COLLECTION_NAME = "vector-db-playground"


def get_qdrant_client():
    global client
    path = "./vector-store"
    if client is None:
        shutil.rmtree(path, ignore_errors=True)
        client = QdrantClient(path=path)
        print("Qdrant client created successfully.")
    return client


def create_collection(client, collection_name=COLLECTION_NAME):
    if not client:
        client = get_qdrant_client()

    if collection_name not in client.get_collections().collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.EUCLID),
        )
        print(f"Collection '{collection_name}' created successfully.")


def upsert_data(client, collection_name=COLLECTION_NAME, payloads=[]):
    if not client:
        client = get_qdrant_client()
    try:
        client.upsert(collection_name=collection_name, wait=True, points=payloads)
        print(f"Data {payloads} upserted successfully to collection '{collection_name}'.")
    except Exception as e:
        print(f"An error occurred during upsert: {e}")


def search_data(client, query_vector, collection_name=COLLECTION_NAME, top_k=2):
    print(
        f"Searching for query vector {query_vector} in collection '{collection_name}' with top_k={top_k}..."
    )
    if not client:
        client = get_qdrant_client()

    try:
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        pprint(f"Search results for query vector {query_vector}: {search_result}")
        return search_result
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return None


def main():
    print("Hello from vector-db-playground!")
    client = get_qdrant_client()
    create_collection(client=client)
    upsert_data(
        client=client,
        payloads=[
            PointStruct(id=1, vector=[0.01, 0.2], payload={"name": "point1"}),
            PointStruct(id=2, vector=[0.3, 0.04], payload={"name": "point2"}),
            PointStruct(id=3, vector=[-0.5, 0.6], payload={"name": "point3"}),
            PointStruct(id=4, vector=[0.7, 0.01], payload={"name": "point4"}),
            PointStruct(id=5, vector=[0.9, -0.10], payload={"name": "point5"}),
        ],
    )

    payloads = [0.9, 0.12]
    res = search_data(client=client, query_vector=payloads)
    pprint(res, indent=2)

    client.close()


if __name__ == "__main__":
    main()
