# Multi-Query RAG System Documentation

## Overview

This project implements a multi-query Retrieval-Augmented Generation (RAG) system for medical discharge summaries. The system processes patient reports, stores them as vector embeddings in a PostgreSQL database with pgvector extension, and generates comprehensive discharge summaries by querying relevant information.

## Architecture

- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **Embeddings**: ClinicalBERT model for medical text embeddings
- **LLM**: OpenAI GPT-4o-mini for text generation
- **Data Processing**: Chunked text processing for large documents

---

## Module: `config.py`

### `load_config(filename="config.ini", section="database")`

**Purpose**: Loads configuration parameters from an INI file for database connections.

**Parameters**:

- `filename` (str, optional): Path to the configuration file. Defaults to "config.ini"
- `section` (str, optional): Configuration section to read. Defaults to "database"

**Returns**:

- `dict`: Dictionary containing configuration key-value pairs

**Raises**:

- `Exception`: If the specified section is not found in the configuration file

**Usage Example**:

```python
config = load_config()
# Returns: {'host': 'localhost', 'database': 'mydb', 'user': 'postgres', ...}
```

**Notes**:

- Expects INI file format with sections like `[database]`
- Used to centralize database connection parameters

---

## Module: `connect.py`

### `connect(config: Dict[str, Any])`

**Purpose**: Establishes a connection to PostgreSQL database using provided configuration.

**Parameters**:

- `config` (Dict[str, Any]): Database connection parameters dictionary

**Returns**:

- `Optional[connection]`: PostgreSQL connection object if successful, None if failed

**Error Handling**:

- Catches `psycopg2.DatabaseError` and general exceptions
- Prints error messages to console
- Returns None on failure

**Usage Example**:

```python
config = load_config()
conn = connect(config)
if conn:
    # Use connection
    cur = conn.cursor()
```

**Notes**:

- Uses context manager for automatic connection handling
- Prints connection status to console

### `main()`

**Purpose**: Entry point that loads configuration and establishes database connection.

**Parameters**: None

**Returns**: None

**Functionality**:

- Loads configuration using `load_config()`
- Attempts database connection using `connect()`

---

## Module: `db.py`

### `enable_pgvector(cur: cursor, conn: connection)`

**Purpose**: Enables and verifies the pgvector extension in PostgreSQL for vector operations.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object

**Returns**: None

**Functionality**:

- Creates pgvector extension if not exists
- Commits changes to database
- Queries and verifies extension installation
- Prints status messages

**Error Handling**:

- Catches and prints any exceptions during extension creation

### `create_vector_table(cur: cursor, conn: connection)`

**Purpose**: Creates the main table for storing patient reports with vector embeddings.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object

**Returns**: None

**Table Schema**:

```sql
patient_reports (
    id SERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    report TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    embedding VECTOR(768) NOT NULL,
    UNIQUE(patient_id, chunk_index)
)
```

**Features**:

- Supports chunked documents with index tracking
- Unique constraint prevents duplicate chunks per patient
- 768-dimensional vectors (ClinicalBERT embedding size)

### `process_csv_to_vector(file_path: str, cur: cursor, conn: connection)`

**Purpose**: Processes CSV file containing patient reports and stores them with embeddings in the database.

**Parameters**:

- `file_path` (str): Path to CSV file containing patient data
- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object

**Returns**: None

**CSV Expected Format**:

- `id`: Unique patient identifier
- `name`: Patient name
- `report`: Medical report text

**Functionality**:

1. Reads CSV using pandas
2. For each patient record:
   - Generates embeddings with chunking
   - Inserts each chunk as separate database row
   - Handles conflicts with UPSERT operation
3. Commits all changes
4. Provides progress feedback

**Error Handling**:

- Catches and prints processing errors
- Continues processing other records if one fails

### `clear_table(cur: cursor, conn: connection, table_name: str = "patient_reports")`

**Purpose**: Removes all data from specified table while preserving structure.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object
- `table_name` (str, optional): Target table name. Defaults to "patient_reports"

**Returns**: None

**Functionality**:

- Deletes all rows from table
- Resets auto-increment counter to 1
- Commits changes

**Error Handling**:

- Rollback on errors
- Prints error messages

### `drop_table(cur: cursor, conn: connection, table_name: str = "patient_reports")`

**Purpose**: Completely removes table structure and data from database.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object
- `table_name` (str, optional): Target table name. Defaults to "patient_reports"

**Returns**: None

**Functionality**:

- Drops table if exists
- Commits changes

**Error Handling**:

- Rollback on errors
- Uses IF EXISTS to prevent errors

### `truncate_table(cur: cursor, conn: connection, table_name: str = "patient_reports")`

**Purpose**: Fast removal of all table data with identity restart.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor object
- `conn` (connection): PostgreSQL connection object
- `table_name` (str, optional): Target table name. Defaults to "patient_reports"

**Returns**: None

**Functionality**:

- Uses TRUNCATE for performance (faster than DELETE on large tables)
- Restarts identity sequence
- Commits changes

**Performance Notes**:

- More efficient than DELETE for large datasets
- Cannot be rolled back in some PostgreSQL configurations

### `main()`

**Purpose**: Main execution function that sets up database and processes initial data.

**Parameters**: None

**Returns**: None

**Workflow**:

1. Load database configuration
2. Establish connection
3. Enable pgvector extension
4. Create vector table
5. Process CSV data
6. Clean up connections

---

## Module: `embedding.py`

### Global Variables

- `model`: SentenceTransformer instance using "medicalai/ClinicalBERT"
- `os.environ["TOKENIZERS_PARALLELISM"] = "false"`: Disables parallelism warnings

### `chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200)`

**Purpose**: Splits large text into overlapping chunks for processing.

**Parameters**:

- `text` (str): Input text to be chunked
- `chunk_size` (int, optional): Maximum characters per chunk. Defaults to 1000
- `overlap` (int, optional): Character overlap between chunks. Defaults to 200

**Returns**:

- `list[str]`: List of text chunks

**Algorithm**:

1. If text ≤ chunk_size, returns original text
2. Creates overlapping chunks with word boundary preservation
3. Attempts to break at spaces in last 20% of chunk
4. Strips whitespace from chunks

**Design Considerations**:

- Overlap ensures context preservation across chunks
- Word boundary detection prevents mid-word breaks
- Handles edge cases for very short texts

### `get_embedding(text: str, additional_data: Optional[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200)`

**Purpose**: Generates embeddings for text with automatic chunking and metadata integration.

**Parameters**:

- `text` (str): Input text for embedding generation
- `additional_data` (Optional[Dict[str, Any]]): Metadata to prepend (patient_id, patient_name)
- `chunk_size` (int, optional): Maximum characters per chunk. Defaults to 1000
- `overlap` (int, optional): Character overlap between chunks. Defaults to 200

**Returns**:

- `list[tuple[str, list[float]]] | None`: List of (chunk_text, embedding) tuples, or None on error

**Functionality**:

1. Prepends metadata if provided
2. Chunks text using `chunk_text()`
3. Generates embedding for each chunk using ClinicalBERT
4. Converts numpy arrays to Python lists
5. Provides progress feedback

**Error Handling**:

- Catches and logs embedding generation errors
- Returns None on failure

**Metadata Format**:

```python
# Example prepended text:
"patient id: 12345 patient name: John Doe [original text]"
```

### `get_single_embedding(text: str)`

**Purpose**: Generates a single embedding for text without chunking.

**Parameters**:

- `text` (str): Input text for embedding

**Returns**:

- `list[float] | None`: Single embedding vector or None on error

**Use Cases**:

- Query embeddings for similarity search
- Short texts that don't require chunking
- Real-time embedding generation

**Error Handling**:

- Catches and logs errors
- Returns None on failure

---

## Module: `retriever.py`

### Global Variables

- `client`: OpenAI client instance
- `sections`: Loaded query sections from JSON file

### Type Definitions

```python
class SectionQuery(TypedDict):
    section_name: str
    queries: List[str]
```

### Class: `Retriever`

#### `__init__(self, cur: cursor, patient_id: int)`

**Purpose**: Initializes retriever with database cursor and patient ID.

**Parameters**:

- `cur` (cursor): PostgreSQL cursor for database operations
- `patient_id` (int): Target patient ID for retrieval

**Attributes**:

- `self.cur`: Database cursor
- `self.patient_id`: Patient identifier
- `self.sections`: Query sections loaded from JSON

#### `get_context(self, query: str, top_k: int = 5)`

**Purpose**: Retrieves relevant context from database using vector similarity search.

**Parameters**:

- `query` (str): Search query
- `top_k` (int, optional): Number of top results to return. Defaults to 5

**Returns**:

- `str | None`: Formatted context string or None if no results/error

**Algorithm**:

1. Generates embedding for input query
2. Performs cosine similarity search in database
3. Filters results by patient ID
4. Orders by similarity score (pgvector `<->` operator)
5. Formats results with patient and chunk information

**SQL Query**:

```sql
SELECT patient_name, report, chunk_index
FROM patient_reports
WHERE patient_id = %s::text
ORDER BY embedding <-> %s::vector
LIMIT %s;
```

**Output Format**:

```
Patient: [Name]
Report chunk [index]: [text]

Patient: [Name]
Report chunk [index]: [text]
...
```

#### `generate_answer(self, context: str, query: str)`

**Purpose**: Generates answers using OpenAI's GPT model based on context and query.

**Parameters**:

- `context` (str): Retrieved context from database
- `query` (str): User's question

**Returns**:

- `str | None`: Generated answer or None on error

**Model Configuration**:

- Model: "gpt-4o-mini"
- Max tokens: 500
- Temperature: Default (controlled randomness)

**System Prompt**:

- Role: Medical assistant
- Task: Generate concise, informative medical answers
- Format: Plain text
- Tone: Professional medical tone

**Error Handling**:

- Returns None if API call fails
- Handles cases where no choices are returned

#### `get_subsection_output(self, query: str)`

**Purpose**: Generates output for individual subsection queries.

**Parameters**:

- `query` (str): Specific medical query

**Returns**:

- `str`: Generated answer or "No answer generated"

**Workflow**:

1. Retrieves context using `get_context()`
2. Generates answer using `generate_answer()`
3. Provides fallback responses for failures

#### `get_section_output(self, section: SectionQuery)`

**Purpose**: Generates comprehensive summary for an entire section.

**Parameters**:

- `section` (SectionQuery): Section with name and queries

**Returns**:

- `str`: Section summary or "No summary generated"

**Approach**:

1. Creates detailed prompt for section summarization
2. Iterates through all queries in section
3. Retrieves context for each query
4. Builds comprehensive prompt with questions and contexts
5. Generates unified summary for entire section

**Prompt Structure**:

```
You are a helpful assistant... for {section_name}

Question: [query1]
Context: [context1]

Question: [query2]
Context: [context2]
...

Please provide a concise and informative summary...
```

#### `generate_discharge_summary(self)`

**Purpose**: Generates complete discharge summary for all sections.

**Parameters**: None

**Returns**:

- `Dict[str, str]`: Dictionary mapping section names to summaries

**Workflow**:

1. Iterates through all predefined sections
2. Generates summary for each section using `get_section_output()`
3. Compiles results into structured dictionary

**Output Structure**:

```python
{
    "history of present illness": "Summary text...",
    "admission": "Summary text...",
    "physical_exam": "Summary text...",
    ...
}
```

### `main()`

**Purpose**: Main execution function demonstrating retriever usage.

**Functionality**:

1. Loads database configuration
2. Establishes connection
3. Creates Retriever instance
4. Generates complete discharge summary
5. Prints formatted results
6. Cleans up connections

---

## Module: `seed.py`

### `main()`

**Purpose**: Database seeding function that sets up fresh database with initial data.

**Parameters**: None

**Returns**: None

**Workflow**:

1. Load database configuration
2. Establish database connection
3. Enable pgvector extension
4. Drop existing table (fresh start)
5. Create new vector table
6. Process CSV data into database
7. Clean up connections

**Use Cases**:

- Initial database setup
- Development environment reset
- Data refresh operations

**Notes**:

- Destructive operation (drops existing data)
- Should be used carefully in production environments

---

## Module: `main.py`

### `main()`

**Purpose**: Main application entry point demonstrating end-to-end system usage.

**Parameters**: None

**Returns**: None

**Functionality**:

1. Loads environment variables
2. Establishes database connection
3. Creates Retriever instance with example patient ID
4. Generates and displays discharge summary
5. Cleans up resources

**Configuration**:

- Example patient ID: 3 (configurable)
- Demonstrates complete workflow from query to summary

**Output**:

- Formatted discharge summary by section
- Console display of all section summaries

---

## Data Flow Architecture

### 1. Data Ingestion

```
CSV File → process_csv_to_vector() → Chunked Embeddings → PostgreSQL
```

### 2. Query Processing

```
User Query → get_single_embedding() → Vector Search → Context Retrieval
```

### 3. Answer Generation

```
Context + Query → OpenAI API → Generated Answer
```

### 4. Summary Generation

```
Multiple Queries → Section Summaries → Complete Discharge Summary
```

## Error Handling Patterns

### Database Operations

- Connection failures handled with None returns
- Transaction rollbacks on errors
- Graceful degradation with error messages

### Embedding Generation

- Model loading errors caught and logged
- Fallback to None returns for failed embeddings
- Progress tracking for chunked operations

### API Calls

- OpenAI API failures handled gracefully
- Fallback responses for generation failures
- Token limit management

## Performance Considerations

### Vector Search Optimization

- pgvector extension for efficient similarity search
- Indexed vector columns for fast retrieval
- Configurable top-k results to limit response size

### Memory Management

- Chunked processing for large documents
- Streaming approaches for large datasets
- Connection pooling for concurrent operations

### Scalability Features

- Patient-specific filtering for multi-tenant scenarios
- Configurable chunk sizes for different document types
- Batch processing capabilities for large datasets

## Configuration Management

### Database Configuration

- Centralized in `config.ini` file
- Environment-specific settings
- Secure credential management

### Model Configuration

- ClinicalBERT for medical domain accuracy
- Configurable embedding dimensions
- Model caching for performance

### API Configuration

- OpenAI API key management through environment variables
- Configurable model parameters
- Rate limiting considerations

## Security Considerations

### Data Privacy

- Patient data handling in compliance with healthcare regulations
- Secure database connections
- Minimal data exposure in error messages

### API Security

- Secure API key storage
- Request validation
- Error message sanitization

### Database Security

- Parameterized queries to prevent SQL injection
- Connection security
- Access control considerations
