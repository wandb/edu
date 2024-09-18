from functools import partial
from typing import Callable, List, Optional

import numpy as np
from blingfire import text_to_sentences
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm

from .embedding import sync_embed

CHUNK_SIZE = 512


def sentence_splitter(text: str) -> List[str]:
    return text_to_sentences(text).split("\n")


def split_into_chunks(
    text: str, length_function: Callable[[str], int], max_tokens: int
) -> List[str]:
    # Split the text into sentences
    sentences = sentence_splitter(text)

    # Get the number of tokens for each sentence
    n_tokens = [length_function("\n" + sentence) for sentence in sentences]

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


class KamradtModifiedChunker:
    """
    A chunker that splits text into chunks of approximately a specified average size based on semantic similarity.

    This implementation is adapted from
    https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/kamradt_modified_chunker.py

    """

    def __init__(
        self,
        avg_chunk_size: int = CHUNK_SIZE,
        min_chunk_size: int = 50,
        embedding_function: Optional[Callable] = None,
        length_function: Optional[Callable] = None,
    ):
        self.splitter = partial(
            split_into_chunks,
            length_function=length_function,
            max_tokens=min_chunk_size,
        )

        self.avg_chunk_size = avg_chunk_size
        if embedding_function is None:
            embedding_function = sync_embed
        self.embedding_function = embedding_function
        self.min_chunk_size = min_chunk_size
        if length_function is None:
            length_function = len
        self.length_function = length_function

    def combine_sentences(self, sentences, buffer_size=1, sep="\n"):
        n = len(sentences)
        for i in range(n):
            start = max(0, i - buffer_size)
            end = min(n, i + buffer_size + 1)

            combined = sep.join(sentences[j]["sentence"] for j in range(start, end))

            sentences[i]["combined_sentence"] = combined

        return sentences

    def calculate_cosine_distances(self, sentences):
        if len(sentences) <= 1:
            # Not enough sentences to calculate distances
            return [], sentences

        combined_sentences = [sentence["combined_sentence"] for sentence in sentences]
        embeddings = self.embedding_function(combined_sentences)
        embedding_matrix = np.array(embeddings)

        if embedding_matrix.shape[0] <= 1:
            # Not enough embeddings to calculate distances
            return [], sentences

        distances = cosine_distances(
            embedding_matrix[:-1], embedding_matrix[1:]
        ).diagonal()

        for i, distance in enumerate(distances):
            sentences[i]["distance_to_next"] = distance

        return distances.tolist(), sentences

    def split_text(self, text):
        """
        Splits the input text into chunks of approximately the specified average size based on semantic similarity.

        Args:
            text (str): The input text to be split into chunks.

        Returns:
            list of str: The list of text chunks.
        """

        sentences_strips = self.splitter(text)
        if len(sentences_strips) <= 1:
            # Return the original text as a single chunk
            return [text]

        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(sentences_strips)
        ]

        sentences = self.combine_sentences(sentences, 3)

        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(
            self.length_function(sentence["sentence"]) for sentence in sentences
        )
        avg_chunk_size = self.avg_chunk_size
        number_of_cuts = total_tokens // avg_chunk_size

        # Define threshold limits
        lower_limit = 0.0
        upper_limit = 1.0

        # Convert distances to numpy array
        distances_np = np.array(distances)

        # Binary search for threshold
        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)

            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold]

        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks


def chunk_document(doc, chunk_size=CHUNK_SIZE):
    from scripts.utils import length_function

    chunker = KamradtModifiedChunker(
        avg_chunk_size=chunk_size, length_function=length_function
    )
    chunks = chunker.split_text(doc["parsed_content"])
    return [
        {
            "cleaned_content": chunk,
            "metadata": {
                "source": doc["metadata"]["source"],
                "parsed_tokens": length_function(chunk),
            },
        }
        for chunk in chunks
    ]


def chunk_documents(docs, chunk_size=CHUNK_SIZE):
    chuker = partial(chunk_document, chunk_size=chunk_size)
    chunked_data = map(chuker, docs)
    chunked_docs = [
        item for sublist in tqdm(chunked_data, total=len(docs)) for item in sublist
    ]
    return chunked_docs
