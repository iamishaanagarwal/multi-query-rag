from multi_query_rag.config import load_config
from multi_query_rag.connect import connect
from multi_query_rag.retriever import Retriever
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
    query = "Generate a discharge summary this patient's medical report."
    patient_id = 3  # Example patient ID, adjust as needed

    # Retrieve context from the database
    retriever = Retriever(cur, patient_id)
    summaries = retriever.generate_discharge_summary()
    print("Generated Discharge Summary:")
    for section_name, summary in summaries.items():
        print(f"\nSection: {section_name}\nSummary: {summary}")

    # Close cursor and connection
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
