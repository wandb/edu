from blingfire import text_to_sentences
from typing import List, Callable

CHUNK_SIZE = 500


def split_into_chunks(
    text: str, tokenize_text: Callable[[str], List[str]], max_tokens: int = CHUNK_SIZE
) -> List[str]:
    # Split the text into sentences
    sentences = text_to_sentences(text).split("\n")

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenize_text("\n" + sentence).tokens) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append("\n".join(chunk))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add any remaining chunk
    if chunk:
        chunks.append("\n".join(chunk))

    return chunks
