from typing import Any, Dict, Optional
from sentence_transformers import SentenceTransformer
import os

# Disable parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("medicalai/ClinicalBERT")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at word boundaries
        if end < len(text):
            last_space = chunk.rfind(" ")
            if (
                last_space > chunk_size * 0.8
            ):  # Only adjust if we find a space in the last 20%
                chunk = chunk[:last_space]
                end = start + last_space

        chunks.append(chunk.strip())
        start = end - overlap

        if start >= len(text):
            break

    return chunks


def get_embedding(
    text: str,
    additional_data: Optional[Dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[list[float]] | None:
    """Get embeddings for text, chunking if necessary"""
    try:
        # Chunk the text
        text_to_embed = text
        if additional_data:
            text_to_embed = f"patient id: {additional_data.get('patient_id', '')} patient name: {additional_data.get('patient_name', '')} {text}"
        chunks = chunk_text(text_to_embed, chunk_size, overlap)
        embeddings = []
        

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            embeddings.append(embedding.tolist())  # Convert numpy array to list
            print(f"Generated embedding for chunk {i+1}/{len(chunks)}")

        return embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None
    
def get_single_embedding(text: str) -> list[float] | None:
    """Get a single embedding for text without chunking"""
    try:
        response = model.encode(text)
        return response.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None