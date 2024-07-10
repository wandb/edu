
import cohere
import os
import weave
import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


client = cohere.AsyncClient(api_key=os.environ["CO_API_KEY"])

@weave.op()
async def evaluate_retriever_using_llm_judge(query: str, passage: str, eval_prompt: str) -> str:
    response = await client.chat(
        message=eval_prompt.format(query=query, document=passage),
        model="command-r-plus",
        temperature=0.0,
        max_tokens=20,
    )
    return response.text


@weave.op()
async def run_retriever_evaluation_using_llm(
    eval_samples: List[Dict[str, Any]],
    retriever: Any,
    eval_prompt: str
) -> List[Dict[str, Any]]:
    scores = []
    for sample in eval_samples:
        query = sample["question"]
        search_results = retriever.search(query, k=5)
        tasks = []
        for result in search_results:
            tasks.append(evaluate_retriever_using_llm_judge(query, result["text"], eval_prompt))
        sample_scores = await asyncio.gather(*tasks)
        parsed_scores = []
        for score in sample_scores:
            try:
                score = json.loads(score)
                parsed_scores.append(score["final_score"])
            except json.JSONDecodeError:
                parsed_scores.append(None)
        scores.append({"query": query, "scores": parsed_scores})
    return scores
