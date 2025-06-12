from multi_query_rag.retriever import get_context, generate_answer
from multi_query_rag.connect import connect, load_config
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())



def main():
    """Main function to run the retriever operations"""
    config = load_config()
    conn = connect(config)

    if not conn:
        raise Exception("Failed to connect to the database")

    cur = conn.cursor()

    # Example query
    query = "Generate a discharge summary for Chloe Fernandez."

    # Retrieve context from the database
    context = get_context(cur, query)

    if not context:
        print("No relevant context found for the query.")
        return

    # print(f"Retrieved context:\n{context}\n")

    # Generate answer using OpenAI
    answer = generate_answer(context, query)
    print(f"Generated answer: {answer}")

    # Close cursor and connection
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
