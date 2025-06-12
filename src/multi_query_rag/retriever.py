from openai import OpenAI
from multi_query_rag.config import load_config
from typing import Dict, Any
from multi_query_rag.connect import connect
from psycopg2.extensions import connection, cursor
from multi_query_rag.db import chunk_text
import os
from dotenv import load_dotenv, find_dotenv

if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

query = "Generate a discharge summary for Chloe Fernandez."


def get_single_embedding(text: str) -> list[float] | None:
    """Get a single embedding for text without chunking"""
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def get_context(cur: cursor, query: str, top_k: int = 5) -> str | None:
    """
    Retrieve context for a given query from the PostgreSQL database.

    Args:
        cur (cursor): The database cursor.
        query (str): The query for which to retrieve context.
        top_k (int): The number of top results to return.

    Returns:
        str: The retrieved context.
    """
    query_embedding = get_single_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding")
        return None

    cur.execute(
        """
        SELECT patient_name, report, chunk_index 
        FROM patient_reports
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """,
        (query_embedding, top_k),
    )

    results = cur.fetchall()
    if results:
        # Format the context with patient info
        context_parts = []
        for patient_name, report, chunk_index in results:
            context_parts.append(
                f"Patient: {patient_name}\nReport chunk {chunk_index}: {report}"
            )
        return "\n\n".join(context_parts)

    return None


def generate_answer(context: str, query: str) -> str | None:
    """
    Generate an answer based on the context and query using OpenAI's API.

    Args:
        context (str): The context retrieved from the database.
        query (str): The user's query.

    Returns:
        str: The generated answer.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant. Generate comprehensive and accurate medical summaries based on the provided patient reports.",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"},
        ],
        max_tokens=500,
    )

    if response.choices:
        return response.choices[0].message.content

    return None


def main():
    config = load_config()
    conn = connect(config)

    if not conn:
        raise Exception("Failed to connect to the database")
    cur = conn.cursor()

    context = get_context(cur, query)
    # print(f"Context retrieved for query '{query}':\n{context}\n")
    if not context:
        print("No relevant context found for the query.")
        return

    print(f"Retrieved context:\n{context}\n")

    answer = generate_answer(context, query)
    print(f"Generated answer: {answer}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
