from langchain.agents import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.utilities import PythonREPL

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

def search_code_examples(file_type="", programming_language="", language="", query=""):

    # https://github.com/wandb/examples/tree/master/colabs#%EF%B8%8F-wb-features
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


def search_developer_docs(query):
    developer_docs = [{"text" : "What is a W&B Run?\nA W&B run is a single unit of computation logged by W&B, representing an atomic element of your project."},
                      
    {"text" : "How do I view a specific run?\nTo view a run, navigate to the W&B App UI, select the relevant project, and then choose the run from the 'Runs' table."},
    
    {"text" : "What are Artifact Outputs?\nArtifact Outputs refer to any artifacts produced by a run."},
    
    {"text" : "How can I do hyperparameter search quickly?\nUse W&B Sweeps to automate hyperparameter search and visualize rich, interactive experiment tracking."}
    ]

    
    return {"developer_docs": developer_docs}

def search_company_information(query):
  
  company_information = [
    {"text" : "Where are Weights & Biases offices?\nWeights & Biases have offices in San Francisco, Berlin, Tokyo, and London."},
    
    {"text" : "Current open positions at Weights & Biases\nCustomer Success Manager, Solution Architect, Success ML Engineer."},
    
    {"text" : "Who are the co-founders of Weights & Biases?\nLukas Biewald, Chris Van Pelt, and Shawn Lewis."}]
  
  return {"company_information": company_information}
    
  
# def search_company_info(query):

tools = [
    {
      "name": "search_code_examples",
      "description": "Searches code examples and tutorials on using Weights & Biases.",
      "parameter_definitions": {
        "file_type": {
          "description": """A FileFormat object that represents an item. The definition of FileFormat looks like the following:
@dataclass
class FileFormat:
    fileformat: str  # one of ["py", "ipynb"]
""",
          "type": "str",
          "required": False
        },
        "language": {
          "description": """A Language object that represents an item. Use this to define the language based on either the language the user is using (either English or Japanese) or if the user is asking examples of a specific language. The definition of Language looks like the following:
@dataclass
class Language:
    language: str  # one of ["en", "ja"]
""",
          "type": "str",
          "required": True
        },
        "query": {
          "description": "The query to search for code examples.",
          "type": "str",
          "required": True
        }
      }
    }, 
    {
      "name": "search_developer_docs",
      "description": "Searches the Weights & Biases developer documentation. Use this tool for queries related to the Weights & Biases API, SDKs, or other developer resources.",
      "parameter_definitions": {
        "query": {
          "description": "The search query.",
          "type": "str",
          "required": True
        }
      }
    }, 
    {
      "name": "search_company_information",
      "description": "Searches the company information about Weights & Biases. Use this tool for general queries about the company.",
      "parameter_definitions": {
        "query": {
          "description": "The search query.",
          "type": "str",
          "required": True
        }
      }
    },
    {
        "name": "analyze_evaluation_results",
        "description": "Generate Python code using the pandas library to analyze evaluation results from a dataframe called `evaluation_results`. The dataframe has columns 'usecase','run','score','temperature','tokens', and 'latency'. You must start with `import pandas as pd` and read a CSV file called `evaluation_results.csv` into the `evaluation_results` dataframe.",
        "parameter_definitions": {
            "code": {
                "description": "Executable Python code",
                "type": "str",
                "required": True
            }
        }
    }
]