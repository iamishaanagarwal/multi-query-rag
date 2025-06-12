from multi_query_rag.config import load_config
from multi_query_rag.connect import connect
from multi_query_rag.db import (
    enable_pgvector,
    create_vector_table,
    process_csv_to_vector,
)


def main():
    """Main function to run the seed operations"""
    config = load_config()
    conn = connect(config)

    if not conn:
        raise Exception("Failed to connect to the database")

    cur = conn.cursor()

    # Enable pgvector extension
    enable_pgvector(cur, conn)

    # Create vector table
    create_vector_table(cur, conn)

    process_csv_to_vector("mock_patient_reports.csv", cur, conn)

    # Close cursor and connection
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
