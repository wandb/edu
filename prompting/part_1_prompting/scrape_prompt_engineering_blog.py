import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def element_to_markdown(element):
    """Convert an HTML element to Markdown."""
    if element.name == 'h1':
        return f"# {element.get_text()}\n"
    elif element.name == 'h2':
        return f"## {element.get_text()}\n"
    elif element.name == 'h3':
        return f"### {element.get_text()}\n"
    elif element.name == 'p':
        return f"{element.get_text()}\n"
    elif element.name == 'ul':
        return '\n'.join(f"- {li.get_text()}" for li in element.find_all('li')) + "\n\n"
    elif element.name == 'ol':
        return '\n'.join(f"{idx + 1}. {li.get_text()}" for idx, li in enumerate(element.find_all('li'))) + "\n\n"
    elif element.name == 'img':
        alt_text = element.get('alt', '')
        img_url = element.get('src', '')
        return f"![{alt_text}]({img_url})\n\n"
    elif element.name == 'a':
        link_text = element.get_text()
        link_url = element.get('href', '')
        return f"[{link_text}]({link_url})\n"
    elif element.name == 'blockquote':
        return f"> {element.get_text()}\n\n"
    elif element.name == 'pre' or element.name == 'code':
        code_text = element.get_text()
        return f"```\n{code_text}\n```\n\n"
    else:
        return ''

def scrape_to_markdown(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Assuming the main content is within an article tag or similar. Adjust the selector as needed.
    main_content = soup.find('article') or soup

    markdown_content = []

    for element in main_content.find_all(True):  # find_all(True) retrieves all tags
        markdown = element_to_markdown(element)
        if markdown:  # Only add non-empty strings
            markdown_content.append(markdown)

    return ''.join(markdown_content)

def main():
    # URL of the page to scrape
    urls = [
        'https://aman.ai/primers/ai/prompt-engineering/',
        'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#instruction-prompting',
    ]

    # Get the Markdown formatted content
    for url in urls:
        print(f"Scraping {url}")
        markdown_content = scrape_to_markdown(url)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split('.')[0]
        if domain.startswith('www'):
            blog = domain[4:]
        else:
            blog = domain
        # Optionally, save the Markdown content to a file
        with open(f"{blog}_prompt_engineering.md", 'w', encoding='utf-8') as file:
            file.write(markdown_content)

        print(f"Markdown content saved to prompt_engineering.md for {url}")

if __name__ == '__main__':
    main()