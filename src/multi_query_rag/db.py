import pandas as pd
from multi_query_rag.config import load_config
from openai import OpenAI
from psycopg2.extensions import connection, cursor
import os
from dotenv import load_dotenv, find_dotenv

from multi_query_rag.connect import connect
from multi_query_rag.embedding import get_embedding, chunk_text

if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

"""
Columns in the CSV file:
- id: Unique identifier for each document
- name: Name of the patient
- report: First report text
"""


def enable_pgvector(cur: cursor, conn: connection):
    """Enable and verify pgvector extension"""
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("pgvector extension enabled successfully")
    except Exception as e:
        print(f"Error enabling pgvector: {e}")

    # Check if extension is enabled
    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    result = cur.fetchone()
    if result:
        print("pgvector extension is installed and enabled")
        print(f"Extension details: {result}")
    else:
        print("pgvector extension is NOT enabled")


def create_vector_table(cur: cursor, conn: connection):
    """Create a table with patient data and vector embeddings"""
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_reports (
                id SERIAL PRIMARY KEY,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                report TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                embedding VECTOR(768) NOT NULL,
                UNIQUE(patient_id, chunk_index)
            );
        """
        )
        conn.commit()
        print("Table 'patient_reports' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")


def process_csv_to_vector(file_path: str, cur: cursor, conn: connection) -> None:
    """Process CSV file and insert data with embeddings into the database"""
    try:
        # Read CSV file using pandas
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            patient_id = str(row["id"])  # Convert to string to match TEXT type
            patient_name = row["name"]
            report_text = row["report"]

            # Generate embeddings for the report text (with chunking)
            embeddings = get_embedding(
                report_text,
                additional_data={
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                },
            )

            if embeddings:

                # Insert each chunk as a separate row
                for chunk_idx, (chunk_text_content, embedding) in enumerate(embeddings):
                    chunk_text_content = (
                        chunk_text_content if chunk_idx < len(embeddings) else ""
                    )

                    cur.execute(
                        """
                        INSERT INTO patient_reports (patient_id, patient_name, report, chunk_index, total_chunks, embedding) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (patient_id, chunk_index) DO UPDATE SET
                        patient_name = EXCLUDED.patient_name,
                        report = EXCLUDED.report,
                        total_chunks = EXCLUDED.total_chunks,
                        embedding = EXCLUDED.embedding;
                    """,
                        (
                            patient_id,
                            patient_name,
                            chunk_text_content,
                            chunk_idx,
                            len(embeddings),
                            embedding,
                        ),
                    )

                print(
                    f"Processed patient {patient_name} (ID: {patient_id}) - {len(embeddings)} chunks"
                )
            else:
                print(f"Failed to generate embedding for patient {patient_name}")

        conn.commit()
        print(f"All data from {file_path} processed successfully")

    except Exception as e:
        print(f"Error processing CSV file: {e}")


def clear_table(
    cur: cursor, conn: connection, table_name: str = "patient_reports"
) -> None:
    """Clear all entries from the specified table"""
    try:
        cur.execute(f"DELETE FROM {table_name};")
        conn.commit()
        print(f"All entries cleared from table '{table_name}'")

        # Optionally, reset the auto-increment counter if using SERIAL
        cur.execute(
            f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), 1, false);"
        )
        conn.commit()
        print(f"Auto-increment counter reset for table '{table_name}'")

    except Exception as e:
        print(f"Error clearing table '{table_name}': {e}")
        conn.rollback()


def drop_table(
    cur: cursor, conn: connection, table_name: str = "patient_reports"
) -> None:
    """Drop the entire table (structure and data)"""
    try:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"Table '{table_name}' dropped successfully")
    except Exception as e:
        print(f"Error dropping table '{table_name}': {e}")
        conn.rollback()


def truncate_table(
    cur: cursor, conn: connection, table_name: str = "patient_reports"
) -> None:
    """Truncate table (faster than DELETE for large tables)"""
    try:
        cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY;")
        conn.commit()
        print(f"Table '{table_name}' truncated successfully")
    except Exception as e:
        print(f"Error truncating table '{table_name}': {e}")
        conn.rollback()


def main():
    """Main function to run the database operations"""
    config = load_config()
    conn = connect(config)

    if not conn:
        raise Exception("Failed to connect to the database")

    cur = conn.cursor()

    # Enable pgvector extension
    enable_pgvector(cur, conn)

    

    # Create vector table
    create_vector_table(cur, conn)

    # Process the CSV file and insert data into the database
    process_csv_to_vector("mock_patient_reports.csv", cur, conn)

    # Close cursor and connection
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
