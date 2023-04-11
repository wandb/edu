# test this script (only works for Hamel Husain's account), replace these with your own values:
# WANDB_RUN_ID=5ornt00u WANDB_RUN_PATH="hamelsmu/cicd_demo/" WANDB_REGISTRY_PATH="hamelsmu/model-registry/CICD_Demo_Registry" python ./client/promote_model.py

import os, wandb
from urllib.parse import urlencode

assert os.getenv('WANDB_API_KEY'), 'You must set the WANDB_API_KEY environment variable'
run_id = os.environ['WANDB_RUN_ID']
run_path = os.environ['WANDB_RUN_PATH']
registry_path = os.environ['WANDB_REGISTRY_PATH']
tag = os.environ.get('WANDB_TAG', 'production candidate')

api = wandb.Api()
run = api.run(f'{run_path}/{run_id}')

art = [a for a in run.logged_artifacts() if a.type == 'model']
if art:
    assert len(art) == 1, 'More then 1 artifact of type model!'
    art[0].link(registry_path, aliases=[tag])

versions = api.artifact_versions('model', registry_path)
latest_model = versions[0]
query = urlencode({'selectionPath': registry_path, 'version': latest_model.version})
registry_url = f'https://wandb.ai/{latest_model.entity}/registry/model?{query}'

if os.getenv('CI'): # is set to `true` in GitHub Actions https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f: # write the output variable REPORT_URL to the GITHUB_OUTPUT file
        print(f'REGISTRY_URL={registry_url}', file=f)


print(f'The model is promoted to this registry: {registry_url} with the tag `{tag}`')
