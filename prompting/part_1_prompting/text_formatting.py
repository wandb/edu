import textwrap
from IPython.display import Markdown, display_markdown, display

class TextWrapperDisplay:
    def __init__(self, text, max_width=80):
        self.text = text
        self.max_width = max_width

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f'TextWrapperDisplay(...)')
        else:
            wrapped_text = textwrap.fill(self.text, width=self.max_width)
            p.text(wrapped_text)

def display_wrapped_text(text, max_width=100):
    display(TextWrapperDisplay(text, max_width))

def escape_xml_tags(text):
    text = text.replace("<", "\n&lt;").replace(">", "&gt;\n")
    return text

def render(md_text, markdown=False):
    if markdown:
        display_markdown(Markdown(escape_xml_tags(md_text)))
    else:
        display_wrapped_text(md_text)