from typing import List, Dict
from pydantic import BaseModel, Field
import csv
from typing import List, Dict
from litellm import completion
import weave
from instructor import from_litellm
import requests
import io
import weave

from set_env import set_env
set_env("OPENAI_API_KEY")
set_env("WANDB_API_KEY")

completion_with_instructor = from_litellm(completion)

class Context(BaseModel):
    content: str = Field(..., description="A relevant excerpt from a document")
    source: str = Field(..., description="The filename of the source document")
    score: float = Field(..., ge=0, le=1, description="A relevance score between 0 and 1")
    relevance: int = Field(..., ge=0, le=2, description="An integer relevance score: 0 (not relevant), 1 (somewhat relevant), or 2 (highly relevant)")
    chunk_index: int = Field(..., description="The index of the chunk in the dataset")

class ContextList(BaseModel):
    contexts: List[Context] = Field(..., description="A list of relevant contexts")

def filter_chunked_data(chunked_data: List[Dict], source_docs: str) -> List[Dict]:
    source_docs_list = [doc.strip() for doc in source_docs.strip('*').split(',')]
    return [chunk for chunk in chunked_data if any(doc in chunk['metadata']['source'] for doc in source_docs_list)]

@weave.op()
def generate_contexts(question: str, answer: str, source_docs: str, filtered_chunks: List[Dict]) -> ContextList:
    prompt = f"""
    Given the following question and answer pair, generate a list of relevant contexts that could have been used to answer the question. Use ONLY the provided chunked documents to inform your selections.

    Question: {question}
    Answer: {answer}
    Source Documents: {source_docs}

    Select up to 5 most relevant context entries from the chunked documents, focusing EXCLUSIVELY on the mentioned source documents. You MUST only use the contexts provided in the available chunks. Include the chunk_index of each selected chunk in your response.

    Important:
    1. Do NOT generate or invent any context that is not present in the provided chunks.
    2. Only select contexts from the source documents mentioned.
    3. If you can't find relevant contexts in the provided chunks, select fewer or no contexts rather than inventing information.
    """
    
    # Add filtered chunks to the prompt
    prompt += "\n\nAvailable chunks:\n"
    for chunk in filtered_chunks:
        prompt += f"Chunk Index: {chunk['metadata']['chunk_index']}\nSource: {chunk['metadata']['source']}\nContent: {chunk['cleaned_content']}\n\n"
    
    contexts = completion_with_instructor.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with finding relevant contexts for question-answer pairs. You must only use the provided chunks and never invent or generate new information."},
            {"role": "user", "content": prompt}
        ],
        response_model=ContextList
    )
    
    return contexts

def clean_csv_content(csv_content: str) -> str:
    """Remove BOM and clean CSV content."""
    # Remove BOM if present
    cleaned_content = csv_content.lstrip('\ufeff')
    return cleaned_content

@weave.op()
def process_csv(file_path: str, chunked_data: List[Dict]) -> List[Dict]:
    processed_data = []
    
    # Fetch the CSV content from the URL
    response = requests.get(file_path)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Clean the CSV content
    cleaned_csv_content = clean_csv_content(response.text)
    csv_file = io.StringIO(cleaned_csv_content)

    reader = csv.DictReader(csv_file)
    for row in reader:
        question = row['Question']
        answer = row['Answer']
        source_docs = row['Source Docs']
        
        filtered_chunks = filter_chunked_data(chunked_data, source_docs)
        contexts = generate_contexts(question, answer, source_docs, filtered_chunks)
        
        processed_item = {
            'question': question,
            'answer': answer,
            'source_docs': source_docs,
            'question_type': row['Question Type'],
            'source_chunk_type': row['Source Chunk Type'],
            'contexts': [context.dict() for context in contexts.contexts]  # Unpack contexts into dicts
        }
        
        processed_data.append(processed_item)
    
    return processed_data

# Usage
if __name__ == "__main__":
    weave.init("rag-course-finance")
    # Load the chunked data
    chunked_data = weave.ref("chunked_data:latest").get().rows
    csv_file_path = 'https://raw.githubusercontent.com/docugami/KG-RAG-datasets/refs/heads/main/sec-10-q/data/v1/qna_data_mini.csv'
    eval_data = process_csv(csv_file_path, chunked_data)

    eval_dataset = weave.Dataset(name="eval_data", rows=eval_data)
    weave.publish(eval_dataset)
