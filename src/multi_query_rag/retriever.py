import json
from typing import Dict, List, TypedDict
from openai import OpenAI
from multi_query_rag.config import load_config
from multi_query_rag.connect import connect
from psycopg2.extensions import cursor
import os
from dotenv import load_dotenv, find_dotenv
from multi_query_rag.embedding import get_single_embedding

if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

query = "Generate a discharge summary for Chloe Fernandez."
class SectionQuery(TypedDict):
    section_name: str
    queries: List[str]

# Loading your JSON
with open('queries.json', 'r') as f:
    sections: List[SectionQuery] = json.load(f)

class Retriever:
    def __init__(self, cur: cursor, patient_id: int ):
        self.cur = cur
        self.patient_id = patient_id
        self.sections = sections

    def get_context(self, query: str, top_k: int = 5) -> str | None:
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

        self.cur.execute(
            """
            SELECT patient_name, report, chunk_index 
            FROM patient_reports
            WHERE patient_id = %s::text
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """,
            (self.patient_id, query_embedding, top_k),
        )

        results = self.cur.fetchall()
        if results:
            # Format the context with patient info
            context_parts = []
            for patient_name, report, chunk_index in results:
                context_parts.append(
                    f"Patient: {patient_name}\nReport chunk {chunk_index}: {report}"
                )
            return "\n\n".join(context_parts)

        return None


    def generate_answer(self, context: str, query: str) -> str | None:
        """
        Generate an answer based on the context and query using OpenAI's API.

        Args:
            context (str): The context retrieved from the database.
            query (str): The user's query.

        Returns:
            str: The generated answer.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant. Your task is to generate concise and informative answers based on the provided context and user queries. Ensure that your responses are relevant to the medical domain and maintain a professional tone. The answers should be in plain text format.",
                },
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"},
            ],
            max_tokens=500,
        )

        if response.choices:
            return response.choices[0].message.content

        return None

    def get_subsection_output(self, query: str):
        """
        Generate outputs for each subsection in a section based on the queries provided.

        Args:
            query (str): The query for which to generate an answer.
            patient_id (int): The ID of the patient for whom the report is being generated.

        Returns:
            str: The generated answer for the subsection.
        """
        context = self.get_context(query)
        answer = self.generate_answer(
            context=context if context else "No context available",
            query=query
        )
        return answer if answer else "No answer generated"

    def get_section_output(self, section: SectionQuery) -> str:
        prompt = f'''You are a helpful assistant being used as a summarizing expert for getting the summary for each section that will be used for creating medical discharge report, you will be presented with a series of questions and context based on which you have to create a summary for {section["section_name"]}'''
        for query in section["queries"]:
            context = self.get_context(query)
            prompt += f'\n\nQuestion: {query}\nContext: {context if context else "No context available"}'
        prompt += '\n\nPlease provide a concise and informative summary based on the above questions and answers. The summary should be relevant to the section and should not include any personal information about the patient. It should be written in a professional and medical tone, suitable for a discharge report.'
        
        output = self.generate_answer(
            context=prompt,
            query=f"Generate a summary for the section: {section['section_name']}"
        )
        return output if output else "No summary generated"

    def generate_discharge_summary(self) -> Dict[str, str]:
        summaries = {}
        for section in self.sections:
            summary = self.get_section_output(section)
            summaries[section["section_name"]] = summary
        return summaries


def main():
    config = load_config()
    conn = connect(config)

    if not conn:
        raise Exception("Failed to connect to the database")
    cur = conn.cursor()

    retriever = Retriever(cur, patient_id=1)

    summaries = retriever.generate_discharge_summary()

    print("Generated Discharge Summary:")
    for section_name, summary in summaries.items():
        print(f"\nSection: {section_name}\nSummary: {summary}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
