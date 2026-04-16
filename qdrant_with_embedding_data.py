import shutil
from pprint import pprint
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
            vectors_config=VectorParams(size=1536, distance=Distance.EUCLID),
        )
        print(f"Collection '{collection_name}' created successfully.")


def upsert_data(client, collection_name=COLLECTION_NAME, payloads=[]):
    if not client:
        client = get_qdrant_client()
    try:
        client.upsert(collection_name=collection_name, wait=True, points=payloads)
        print(f"Data upserted successfully to collection '{collection_name}'.")
    except Exception as e:
        print(f"An error occurred during upsert: {e}")


def search_data(client, query_vector, collection_name=COLLECTION_NAME, query_filter=None, top_k=2):
    print(
        f"Searching for query vector in collection '{collection_name}' with top_k={top_k}..."
    )
    if not client:
        client = get_qdrant_client()

    try:
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter
        )
        pprint(f"Search results: {search_result}")
        return search_result
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return None


def main():
    print("Hello from vector-db-playground!")
    openai_client = OpenAI()
    client = get_qdrant_client()
    create_collection(client=client)
    interpreted_languages = ["Python", "JavaScript", "Ruby", "PHP", "Perl", "Lua", "R", "MATLAB", "Groovy"]
    compiled_languages = ["C", "C++", "Go", "Rust", "Swift", "Kotlin", "Dart", "Scala", "Haskell"]
    embedded_interpreted_languages = []
    embedded_compiled_languages = []

    for i, language in enumerate(interpreted_languages):
        embedding = openai_client.embeddings.create(input=language, model="text-embedding-3-small")
        embedded_interpreted_languages.append(PointStruct(id=uuid4(), vector=embedding.data[0].embedding, payload={"type": "interpreted", "language": interpreted_languages[i]}))

    for i, language in enumerate(compiled_languages):
        embedding = openai_client.embeddings.create(input=language, model="text-embedding-3-small")
        embedded_compiled_languages.append(PointStruct(id=uuid4(), vector=embedding.data[0].embedding, payload={"type": "compiled", "language": compiled_languages[i]}))

    upsert_data(client, payloads=embedded_interpreted_languages)
    upsert_data(client, payloads=embedded_compiled_languages)

    vector = openai_client.embeddings.create(input="C#", model="text-embedding-3-small").data[0].embedding
    res = search_data(client=client, query_vector=vector, top_k=3)
    pprint(res, indent=4, width=40, depth=2)

    print("\nWith filtering data")
    res = search_data(client=client, 
                      query_vector=vector, 
                      top_k=3,
                      query_filter=Filter(
                            must=FieldCondition(
                                key="type",
                                match=MatchValue(value="interpreted")
                            )
                      ))
    pprint(res, indent=4, width=40, depth=2)

    client.close()


if __name__ == "__main__":
    main()
