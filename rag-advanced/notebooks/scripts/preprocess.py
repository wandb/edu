"""
This module provides functionality to preprocess text data for tokenization and embedding.
"""
import frontmatter
import markdown
from bs4 import BeautifulSoup

from .utils import get_special_tokens_set


def convert_contents_to_text(contents: str) -> str:
    """
    Converts the given markdown content to plain text.

    Args:
        contents: A string containing the markdown content.

    Returns:
        A string containing the plain text extracted from the markdown content.
    """
    _, content = frontmatter.parse(contents)
    markdown_document = markdown.markdown(
        content,
        extensions=[
            "toc",
            "pymdownx.extra",
            "pymdownx.blocks.admonition",
            "pymdownx.magiclink",
            "pymdownx.blocks.tab",
            "pymdownx.pathconverter",
            "pymdownx.saneheaders",
            "pymdownx.striphtml",
            "pymdownx.highlight",
            "pymdownx.pathconverter",
            "pymdownx.escapeall",
        ],
    )
    soup = BeautifulSoup(markdown_document, "html.parser")

    def remove_urls_a_tags_hrefs(soup):
        """
        Removes URLs, <a> tags, and <img> tags from the BeautifulSoup object.

        Args:
            soup: A BeautifulSoup object containing the HTML content.

        Returns:
            A BeautifulSoup object with URLs, <a> tags, and <img> tags removed.
        """
        for a_tag in soup.find_all("a"):
            a_tag.replace_with(a_tag.text)

        for img_tag in soup.find_all("img"):
            img_tag.decompose()

        return soup

    soup = remove_urls_a_tags_hrefs(soup)

    def remove_javascript_import_statements(soup):
        """
        Removes JavaScript import statements from the BeautifulSoup object.

        Args:
            soup: A BeautifulSoup object containing the HTML content.

        Returns:
            A BeautifulSoup object with JavaScript import statements removed.
        """
        for p in soup.find_all("p"):
            if p.text.strip().startswith("import") and ";" in p.text:
                p.decompose()
        return soup

    soup = remove_javascript_import_statements(soup)

    return soup.get_text()


def make_text_tokenization_safe(
    content: str, special_tokens_set: set = get_special_tokens_set()
) -> str:
    """
    Makes the text safe for tokenization by removing special tokens.

    Args:
        content: A string containing the text to be processed.
        special_tokens_set: A set of special tokens to be removed from the text.

    Returns:
        A string with the special tokens removed.
    """

    def remove_special_tokens(text: str) -> str:
        """
        Removes special tokens from the given text.

        Args:
            text: A string representing the text.

        Returns:
            The text with special tokens removed.
        """
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    cleaned_content = remove_special_tokens(content)
    return cleaned_content
