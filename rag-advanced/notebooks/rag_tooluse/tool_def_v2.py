from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL

def search_developer_docs(query: str) -> dict:
    
    developer_docs = [{"text" : "What is a W&B Run?\nA W&B run is a single unit of computation logged by W&B, representing an atomic element of your project."},
                      
    {"text" : "How do I view a specific run?\nTo view a run, navigate to the W&B App UI, select the relevant project, and then choose the run from the 'Runs' table."},
    
    {"text" : "What are Artifact Outputs?\nArtifact Outputs refer to any artifacts produced by a run."},
    
    {"text" : "How can I do hyperparameter search quickly?\nUse W&B Sweeps to automate hyperparameter search and visualize rich, interactive experiment tracking."}
    ]

    return {"developer_docs": developer_docs}
    
def search_internet(query: str) -> dict:
    tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )
    documents = tool.invoke({"query": query})
    
    return {"documents": documents}

def search_code_examples(query: str, file_type: str = None, programming_language: str = None, language: str = None) -> dict:
    
    code_examples = [
        {"content": "Selecting Hyperparameters with Sweeps (Keras)", "file_type": "py", "language": "en"},
        {"content": "Interactive W&B Charts Inside Jupyter", "file_type": "ipynb", "language": "en"},
        {"content": "ロギングメディア", "file_type": "py", "language": "ja"},
        {"content": "テーブルを使用してデータセットと予測を視覚化する", "file_type": "ipynb", "language": "ja"},
        {"content": "Model/Data Versioning with Artifacts (PyTorch)", "file_type": "ipynb", "language": "en"},
        {"content": "Create a hyperparameter search with W&B PyTorch integration", "file_type": "ipynb", "language": "en"},
        {"content": "Get started with W&B Weave", "file_type": "ipynb", "language": "en"}
    ]
    
    filtered_code_examples = [
        item for item in code_examples
        if (item['file_type'] == file_type if file_type else True) and
        (item['programming_language'] == programming_language if programming_language else True) and
        (item['language'] == language if language else True)
    ]
    
    return {"code_examples": filtered_code_examples}

python_repl = PythonREPL()
python_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)
python_tool.name = "python_interpreter"


class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
    
    
python_tool.args_schema = ToolInput


def analyze_evaluation_results(code: str) -> dict:
    """
    Function to run given python code
    """
    input_code = ToolInput(code=code)
    return {'python_answer': python_tool.func(input_code.code)}

### -------------------------------------------------------------------------------------------
search_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_developer_docs",
            "description": "Searches the Weights & Biases developer documentation. Use this tool for queries related to the Weights & Biases API, SDKs, or other developer resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet. Use this tool for general queries that may not be specific to Weights & Biases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search the internet with."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code_examples",
            "description": "Searches code examples and tutorials on using Weights & Biases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "description": """The file format of the code examples to search for. Possible enum values: py, ipynb, null."""
                    },
                    "language": {
                        "type": "string",
                        "description": """The language to choose based on either the language the user is using (either English or Japanese) or if the user is asking examples of a specific language. Possible enum values: en, ja, null."""
                    },
                    "query": {
                        "type": "string",
                        "description": "The query to search for code examples."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

analysis_tool = [
    {
        "type": "function",
        "function": {
            "name": "analyze_evaluation_results",
            "description": "Generate Python code using the pandas library to analyze evaluation results from a dataframe called `evaluation_results`. The dataframe has columns 'usecase','run','score','temperature','tokens', and 'latency'. You must start with `import pandas as pd` and read a CSV file called `evaluation_results.csv` into the `evaluation_results` dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Executable Python code"
                    }
                },
                "required": ["code"]
            }
        }
    }
]