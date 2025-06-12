import psycopg2
from psycopg2.extensions import connection
from multi_query_rag.config import load_config
from typing import Optional, Dict, Any


def connect(config: Dict[str, Any]) -> Optional[connection]:
    """Connect to the PostgreSQL database server"""
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print("Connected to the PostgreSQL server.")
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def main():
    """Main function to load config and connect to the database"""
    config = load_config()
    connect(config)


if __name__ == "__main__":
    main()
