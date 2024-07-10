import frontmatter
import markdown
from bs4 import BeautifulSoup
import requests
import re

def convert_contents_to_text(contents: str) -> str:
    _, content = frontmatter.parse(contents)
    # use some extensions to convert the markdown to html
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
            "pymdownx.escapeall"
        ],
    )
    soup = BeautifulSoup(markdown_document, "html.parser")
    def remove_urls_a_tags_hrefs(soup):
        # For hyperlinks, keep the text but remove the link
        for a_tag in soup.find_all('a'):
            a_tag.replace_with(a_tag.text)
        
        # Remove all images
        for img_tag in soup.find_all('img'):
            img_tag.decompose()
        
        return soup

    # Use the function as before
    soup = remove_urls_a_tags_hrefs(soup)

    def remove_javascript_import_statements(soup):
        for p in soup.find_all('p'):
            if p.text.strip().startswith('import') and ';' in p.text:
                p.decompose()
        return soup
    soup = remove_javascript_import_statements(soup)

    return soup.get_text()

def get_special_tokens_set(tokenizer_url):
    # https://docs.cohere.com/docs/tokens-and-tokenizers
    response = requests.get(tokenizer_url)
    return set([tok["content"] for tok in response.json()["added_tokens"]])

def make_text_tokenization_safe(content: str, special_tokens_set: set) -> str:
    def remove_special_tokens(text: str) -> str:
        """Removes special tokens from the given text.

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

# Note: The 'tokenizers' dictionary and 'special_tokens_set' variable are not defined in this script.
# You'll need to define them or pass them as arguments when using these functions.
