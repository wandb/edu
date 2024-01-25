from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, # for exponential backoff
)

client = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)



