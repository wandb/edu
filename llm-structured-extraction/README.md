<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

# LLM Engineering: Steering Language Models

This repository contains materials for our [LLM Engineering: Steering Language Models](https://www.wandb.courses/courses/steering-language-models) course. 

Improve your LLM engineering skills with Jason Liu, an LLM charmer and the author of Instructor library. Learn everything about structured output handling, function calling, complex validations with Pydantic and more in this short and helpful course!

## 🚀 [Enroll for free](https://www.wandb.courses/courses/steering-language-models)

## What you'll learn

In this course you'll experiment with a list of notebooks that walk you through some simple concepts in Instructor. We start small and then work our way up to more complex and tricky implementations using the library.

1. `Introduction` - This is a quick walkthrough some of the benefits of Pydantic and how the Instructor Library integrates nicely with Pydantic with `instructor.patch()`

2. `Tips` - Quick demonstration of how to use enums, `Pydantic` models and structured prompting to get specific output formats

3. `Applications Rag`: Learn how to generate nested models with `Pydantic` by rewriting user queries

4. `Validation` : Learn how to use Pydantic's inbuilt validators to perform more complex validation and checks on the outputs of your functions

## Installation

We utilise the Graphviz package in this tutorial series. If you don't have it on hand, you should download it. Mac users can do so by running `brew install graphviz` while Linux users can try `sudo apt install graphviz` ( modify to your system specific package manager). Here is a link to their official [documentation](https://graphviz.org/download/)

If you're encountering an error like the following when trying to run graphviz after installing it, just restart the notebook and verify you've got graphviz installed by running `dot -v` in your shell.

```
Command '[PosixPath('dot'), '-Kdot', '-Tsvg']' died with <Signals.SIGKILL: 9>.
```

Here are the steps to start running the notebooks

1. Create a virtual environment

```
python3 -m venv .venv
source .venv .venv/bin/activate
```

2. Install the dependencies

```
pip3 install -r requirements.txt
```

3. Add the virtual environment to Jupyter notebook

```
python -m ipykernel install --user --name=instructor-env
```

4. Add OpenAI API Key into your shell by running the following command. This will be set for as long as the shell is open.

```
export OPENAI_API_KEY=<api key goes here>
```

5. Start Jupyter lab

```
jupyter lab
```
