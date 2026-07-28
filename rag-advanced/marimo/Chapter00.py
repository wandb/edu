# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cohere>=5.13.6",
#     "marimo>=0.23.3",
#     "weave>=0.51.27",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo

    return mo, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter 0: Setup

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter00.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter00.py)

    <!--- @wandbcode{rag-course-00} -->

    Let's check our setup for this course.

    This is a [marimo](https://marimo.io) notebook: the required packages are declared
    inline at the top of the file and are installed automatically when you open the
    notebook with `uvx marimo edit --sandbox Chapter00.py` or on
    [molab](https://molab.marimo.io).

    ### 🎉 Free Cohere API key

    Cohere offers a free tier: signing up at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) automatically gives you a free **trial API key** — no credit card required.

    Keep the trial key limits in mind: chat models are rate-limited to **20 requests/minute**, and a trial key allows **1,000 API calls/month** in total. That is enough for this course if you work through the notebooks one at a time — the evaluation chapters make many LLM calls, so we keep request concurrency low to stay under the limit.

    Alternatively if you have a Cohere production API key feel free to proceed. :)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup Weave

    The cell below will ask for your W&B API key if it can't find existing credentials.
    You can get your API key by heading over to https://wandb.ai/authorize.
    """)
    return


@app.cell
def _(mo, os):
    def _saved_wandb_api_key():
        """Return a W&B API key from ~/.netrc, if the user is already logged in."""
        try:
            import netrc

            auth = netrc.netrc().authenticators("api.wandb.ai")
            return auth[2] if auth else None
        except Exception:
            return None

    wandb_key_found = bool(os.environ.get("WANDB_API_KEY") or _saved_wandb_api_key())
    wandb_key_input = mo.ui.text(
        label="W&B API key ([wandb.ai/authorize](https://wandb.ai/authorize))",
        kind="password",
    )
    (
        mo.callout("W&B credentials found — you are ready to go.", kind="success")
        if wandb_key_found
        else wandb_key_input
    )
    return wandb_key_found, wandb_key_input


@app.cell
def _(mo, os, wandb_key_found, wandb_key_input):
    mo.stop(
        not (wandb_key_found or wandb_key_input.value),
        mo.callout("Enter your W&B API key above to initialize Weave.", kind="warn"),
    )
    if wandb_key_input.value:
        os.environ["WANDB_API_KEY"] = wandb_key_input.value

    import weave

    weave_client = weave.init("rag-course")
    return (weave_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Setup Cohere

    The cell below will ask for your Cohere API key if it isn't already set in the environment.
    """)
    return


@app.cell
def _(mo, os):
    cohere_key_input = mo.ui.text(
        label="Cohere API key ([dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys))",
        kind="password",
    )
    (
        mo.callout("Cohere API key found in the environment.", kind="success")
        if os.environ.get("COHERE_API_KEY")
        else cohere_key_input
    )
    return (cohere_key_input,)


@app.cell
def _(cohere_key_input, mo, os):
    cohere_api_key = os.environ.get("COHERE_API_KEY") or cohere_key_input.value
    mo.stop(
        not cohere_api_key,
        mo.callout("Enter your Cohere API key above to continue.", kind="warn"),
    )
    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["CO_API_KEY"] = cohere_api_key
    # Cohere's free trial keys allow only 20 chat requests/minute; keep Weave's
    # evaluation concurrency at 1 so parallel submissions don't hit 429 errors.
    os.environ.setdefault("WEAVE_PARALLELISM", "1")

    import cohere

    cohere_client = cohere.ClientV2(api_key=cohere_api_key)
    return (cohere_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A simple-turn chat with Cohere's command-r-plus-08-2024
    """)
    return


@app.cell
def _(mo):
    chat_button = mo.ui.run_button(label="Ask Cohere: What is RAG?")
    chat_button
    return (chat_button,)


@app.cell
def _(chat_button, cohere_client, mo, weave_client):
    mo.stop(
        not chat_button.value or weave_client is None,
        mo.md("*Click the button above to send the question to `command-r-plus-08-2024`.*"),
    )
    response = cohere_client.chat(
        messages=[
            {"role": "user", "content": "What is retrieval augmented generation (RAG)?"}
        ],
        model="command-r-plus-08-2024",
        temperature=0.1,
        max_tokens=2000,
    )
    mo.md(response.message.content[0].text)
    return


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"Let's head over to [**your Weave traces**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/traces) to check out the generated response."
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
