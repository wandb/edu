from types import SimpleNamespace

TEAM = None
PROJECT = "llmapps"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    vector_store_artifact="darek/llmapps/vector_store:latest",
    hyde_prompt_artifact="parambharat/wandb_docs_bot/hyde_prompt:latest",
    chat_prompt_artifact="parambharat/wandb_docs_bot/system_prompt:latest",
    model_name="gpt-3.5-turbo",
    eval_model="gpt-3.5-turbo",
    eval_artifact="darek/llmapps/generated_examples:v0",
)

wandb_config = {"project": PROJECT}