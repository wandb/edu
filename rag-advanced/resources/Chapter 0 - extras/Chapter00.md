## Chapter 0: Setup

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter00.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-00} -->

Let's install the required packages and check our setup for this course.

### 🎉 Free Cohere API key

Before you run this colab notebook, head over to this [link to redeem a free Cohere API key](https://docs.google.com/forms/d/e/1FAIpQLSc9x4nV8_nSQvJnaINO1j9NIa2IUbAJqrKeSllNNCCbMFmCxw/viewform?usp=sf_link).

Alternatively if you have a Cohere API key feel free to proceed. :)


```
!pip install -qq weave cohere
```

## 1. Setup Weave


The code cell below will prompt you to put in a W&B API key. You can get your API key by heading over to https://wandb.ai/authorize.


```
# import weave
import weave

# initialize weave client
weave_client = weave.init("rag-course")
```

## 2. Setup Cohere

The code cell below will prompt you to put in a Cohere API key.


```
import getpass

import cohere

cohere_client = cohere.ClientV2(
    api_key=getpass.getpass("Please enter your COHERE_API_KEY")
)
```

## A simple-turn chat with Cohere's command-r-plus


```
response = cohere_client.chat(
    messages=[
        {"role": "user", "content": "What is retrieval augmented generation (RAG)?"}
    ],
    model="command-r-plus",
    temperature=0.1,
    max_tokens=2000,
)
```

Let's head over to the weave URL to check out the generated response.
