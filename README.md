## Setup Instructions (Using UV)

### Prerequisites

- Python 3.13 or higher
- [UV package manager](https://docs.astral.sh/uv/) installed
- OpenAI API key
- PostgreSQL database (Supabase recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/iamishaanagarwal/multi-query-rag.git
   cd "multi-query rag"
   ```

2. **Install dependencies**

   ```bash
   uv sync  # Creates virtual environment and installs all dependencies
   ```

3. **Configure environment variables**

   Create `.env` file:

   ```env
   OPENAI_API_KEY=your-openai-api-key
   DATABASE_HOST=your-database-host
   DATABASE_PORT=5432
   DATABASE_NAME=postgres
   DATABASE_USER=your-username
   DATABASE_PASSWORD=your-password
   ```

4. **Configure database connection**

   Create `config.ini`:

   ```ini
   [database]
   host=your-database-host
   port=5432
   database=your-database-name
   user=your-username
   password=your-password
   ```

5. **Initialize database and run**
   ```bash
   uv run rag-seed   # Setup database and seed data
   uv run rag-query  # Run test query
   ```

### Available Commands

- `uv run rag-seed`: Initialize database with sample data
- `uv run rag-query`: Run RAG query system
- `uv sync`: Install/update dependencies
- `uv add <package>`: Add new dependency

### Development Commands

```bash
uv add <package>        # Add new dependency
uv remove <package>     # Remove dependency
uv tree                 # Show dependency tree
uv run python script.py # Run Python script
```
