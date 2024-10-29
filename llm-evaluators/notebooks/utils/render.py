from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Any, List, Optional
from rich.markdown import Markdown

def print_dialogue_data(
    weave_data: Any,
    indexes_to_show: Optional[List[int]] = None,
    entry_index: int = 0
) -> None:
    """
    Print dialogue data in a formatted way using rich.
    
    Args:
        weave_data: WeaveList containing dialogue data
        indexes_to_show: List of indexes to display (defaults to [0, 1] for input/output)
        entry_index: Index of the dialogue entry to show (defaults to 0)
    """
    console = Console()
    
    # Default indexes if none specified (input and output)
    if indexes_to_show is None:
        indexes_to_show = [0, 1]
    
    # Labels for each index in the tuple
    index_labels = [
        "Input",
        "Output",
        "Annotation",
        "Criteria Annotation",
        "Note",
        "Task Description",
        "Metric Details"
    ]
    
    try:
        # Get the entry from WeaveList
        entry = weave_data[entry_index]
        
        # Print each requested index
        for idx in indexes_to_show:
            if idx >= len(entry):
                console.print(f"[red]Index {idx} is out of range")
                continue
                
            # Get the data and label
            data = entry[idx]
            label = index_labels[idx]
            
            # Handle different data types
            if isinstance(data, dict):
                # Format dictionary contents with one item per line
                content = "\n".join(f"{k}: {v}" for k, v in data.items())
            elif isinstance(data, (int, float, bool)):
                content = str(data)
            elif data is None:
                content = "None"
            else:
                content = str(data)
            
            # Create title with formatting
            title = Text(f" {label} ", style="bold cyan")
            
            # Create and print panel
            panel = Panel(
                content,
                title=title,
                border_style="blue",
                padding=(1, 2),
                expand=True
            )
            console.print(panel)
            console.print()  # Add spacing between panels
                
    except (IndexError, KeyError) as e:
        console.print(f"[red]Error accessing data: {str(e)}")

def display_prompt(prompt_text: str) -> None:
    """
    Renders a prompt nicely formatted in the terminal using rich markdown.
    
    Args:
        prompt_text (str): The prompt text to display
    """
    console = Console()
    md = Markdown(prompt_text)
    console.print(md)