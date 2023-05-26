import argparse
import json
import logging
import os
import pathlib
from typing import Dict, List, Union, Optional

import langchain
import pandas as pd
import tiktoken
import wandb
from langchain import LLMChain
from langchain.vectorstores import Chroma
from langchain.cache import SQLiteCache
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import (
    MarkdownTextSplitter,
    TokenTextSplitter,
)
from tqdm import tqdm
from prompts import load_hyde_prompt

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)

def find_md_files(directory: str = "docs_sample") -> List[Document]:
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    md_files.append(Document(page_content=f.read(), metadata={"source": file_path}))
    return md_files

class DocumentationDatasetLoader:
    """Loads the documentation dataset
    Usage:
    ```
    loader = DocumentationDatasetLoader()
    documents = loader.load()
    # save to disk
    loader.save_to_disk(path)
    # load from disk
    loader.load_from_disk(path)
    ```
    """

    def __init__(
        self,
        documentation_dir: str = "docs_sample",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ):
        """
        :param documentation_dir: The directory containing the documentation from wandb/docodile
        :param chunk_size: The size of the chunks to split the text into using the `TokenTextSplitter`
        :param chunk_overlap: The amount of overlap between chunks of text using the `TokenTextSplitter`
        :param encoding_name: The name of the encoding to use when splitting the text using the `TokenTextSplitter`
        """
        self.documentation_dir = documentation_dir
        self.encoding_name = encoding_name
        self.documents = []
        self.md_text_splitter = MarkdownTextSplitter()
        self.token_splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            allowed_special={"<|endoftext|>"},
        )


    def load_documentation_documents(self, docs_dir: str) -> List[Document]:
        """
        Loads the documentation documents from the specified repository
        :param docs_dir: The directory containing the documentation
        :return: A list of `Document` objects
        """
        documents = find_md_files(directory=docs_dir)
        document_sections = self.md_text_splitter.split_documents(documents)
        document_sections = self.token_splitter.split_documents(document_sections)

        return document_sections

    def load(self) -> List[Document]:
        """
        Loads the documentation
        :return: A list of `Document` objects
        """
        self.documents = []
        if self.documentation_dir and os.path.exists(self.documentation_dir):
            self.documents.extend(
                self.load_documentation_documents(docs_dir=self.documentation_dir)
            )
        else:
            logger.warning(
                f"Documentation directory {self.documentation_dir} does not exist. Not loading documentation."
            )
        print(f"Loaded {len(self.documents)} documents")
        return self.documents

    def save_to_disk(self, path: str) -> None:
        """
        Saves the documents to disk as a jsonl file
        :param path: The path to save the documents to
        """
        with open(path, "w") as f:
            for document in self.documents:
                line = json.dumps(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    }
                )
                f.write(line + "\n")

    @classmethod
    def load_from_disk(cls, path: str) -> "DocumentationDatasetLoader":
        """
        Loads the jsonl documents from disk into a `DocumentationDatasetLoader`
        :param path: The path to the jsonl file containing the documents
        :return: A `DocumentationDatasetLoader` object
        """
        loader = cls()
        with open(path, "r") as f:
            for line in f:
                document = json.loads(line)
                loader.documents.append(Document(**document))
        return loader


class DocumentStore:
    """
    A class for storing and retrieving documents using Chroma and OpenAI embeddings
    """

    base_embeddings = OpenAIEmbeddings()

    def __init__(
        self,
        documents: List[Document],
        use_hyde: bool = True,
        hyde_prompt: Optional[Union[ChatPromptTemplate, str]] = None,
        temperature: float = 0.7,
        path: str = "data/index",
    ):
        """
        :param documents: List of documents to store in the document store
        :param use_hyde: Whether to use the hypothetical document embeddings when embedding documents
        :param hyde_prompt: The prompt to use for the hypothetical document embeddings
        :param temperature: The temperature to use for the hypothetical document embeddings
        """
        self.documents = documents
        self.use_hyde = use_hyde
        self.hyde_prompt = hyde_prompt
        self._embeddings = None
        self._db_store = None
        self.temperature = temperature
        self.path = path

    def embeddings(self) -> Union[Chain, Embeddings]:
        """
        Returns the embeddings to use for the document store
        :return:
        """
        if self._embeddings is None:
            if self.use_hyde:
                if isinstance(self.hyde_prompt, ChatPromptTemplate):
                    prompt = self.hyde_prompt
                elif isinstance(self.hyde_prompt, str) and os.path.isfile(
                    self.hyde_prompt
                ):
                    prompt = load_hyde_prompt(self.hyde_prompt)
                else:
                    prompt = load_hyde_prompt()
                self._embeddings = HypotheticalDocumentEmbedder(
                    llm_chain=LLMChain(
                        llm=ChatOpenAI(temperature=self.temperature), prompt=prompt
                    ),
                    base_embeddings=self.base_embeddings,
                )
            else:
                self._embeddings = self.base_embeddings
        return self._embeddings

    def create_index(
        self,
    ) -> Chroma:
        """
        Creates a Chroma index from documents
        :return: A `Chroma` object
        """

        self._db_store = Chroma.from_documents(
            self.documents, embedding=self.embeddings(),
            persist_directory=self.path,
        )
        return self._db_store

    @property
    def vector_store(self) -> Chroma:
        """
        Returns the Chroma index
        :return: A `Chroma` object
        """
        if self._db_store is None:
            self.create_index()
        return self._db_store

    def save_to_disk(self) -> None:
        """
        Saves the Chroma index to disk
        :param path: The directory to save the Chroma index to
        """
        self.vector_store.persist()

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        use_hyde: bool = True,
        hyde_prompt: Optional[Union[ChatPromptTemplate, str]] = None,
        temperature: float = 0.7,
    ) -> "DocumentStore":
        """
        Loads the `DocumentStore` from disk
        :param path: The directory the Chroma index
        :param use_hyde: Whether to use the hypothetical document embeddings when embedding documents
        :param hyde_prompt: The prompt to use for the hypothetical document embeddings
        :param temperature: The temperature to use for the hypothetical document embeddings
        :return: A `DocumentStore` object
        """
        cls.use_hyde = use_hyde
        cls.hyde_prompt = hyde_prompt
        cls.temperature = temperature
        cls._embeddings = None
        cls._db_store = Chroma(persist_directory=path, embedding_function=cls.embeddings(cls))
        return cls._db_store


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        required=True,
        help="The directory containing the wandb documentation",
    )
    parser.add_argument(
        "--documents_file",
        type=str,
        default="data/documents.jsonl",
        help="The path to save or load the documents to/from",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="data/index",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--hyde_prompt",
        type=str,
        default=None,
        help="The path to the hyde prompt to use",
    )
    parser.add_argument(
        "--use_hyde",
        action="store_true",
        help="Whether to use the hypothetical document embeddings",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="The temperature to use for the hypothetical document embeddings",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)

    if not os.path.isfile(args.documents_file):
        loader = DocumentationDatasetLoader(
            documentation_dir=args.docs_dir,
        )
        documents = loader.load()
        loader.save_to_disk(args.documents_file)
    else:
        loader = DocumentationDatasetLoader.load_from_disk(args.documents_file)
        documents = loader.documents

    documents_artifact = wandb.Artifact("docs_dataset", type="dataset")
    documents_artifact.add_file(args.documents_file)
    run.log_artifact(documents_artifact)
    if not os.path.isdir(args.vector_store):
        document_store = DocumentStore(
            documents=documents,
            use_hyde=args.use_hyde,
            hyde_prompt=args.hyde_prompt,
            temperature=args.temperature,
            path=args.vector_store,
        )
        document_store.save_to_disk()
    else:
        document_store = DocumentStore.load_from_disk(
            args.vector_store,
            use_hyde=args.use_hyde,
            hyde_prompt=args.hyde_prompt,
            temperature=args.temperature,
        )
    index_artifact = wandb.Artifact("vector_store", type="search_index")
    index_artifact.add_dir(args.vector_store)
    run.log_artifact(index_artifact)

    if args.hyde_prompt is not None and os.path.isfile(args.hyde_prompt):
        hyde_prompt_artifact = wandb.Artifact("hyde_prompt", type="prompt")
        hyde_prompt_artifact.add_file(args.hyde_prompt)
        run.log_artifact(hyde_prompt_artifact)

    run.finish()


if __name__ == "__main__":
    main()