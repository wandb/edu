# test this script (only works for Hamel Husain's account), replace these with your own values:
# REPO="hamelsmu/wandb-cicd" PR_NUM="16" WANDB_RUN_ID=5ornt00u REGISTRY_URL="https://wandb.ai/hamelsmu/registry/model?selectionPath=hamelsmu%2Fmodel-registry%2FCICD_Demo_Registry&view=membership&tab=overview&version=test" python ./client/deployment.py

import os
from ghapi.core import GhApi

owner,repo = os.environ['REPO'].split('/')
run_id = os.environ['WANDB_RUN_ID']
pr_num = os.environ['PR_NUM']
registry_url = os.environ['REGISTRY_URL']
environment = os.getenv('DEPLOY_ENVIRONMENT', 'staging')

gapi = GhApi(owner=owner, repo=repo)
branch_name = gapi.pulls.get(pr_num).head.ref


deploy = gapi.repos.create_deployment(ref=branch_name,
                                       environment=environment,
                                       auto_merge=False,
                                       payload={'run_id': run_id, 'registry_url':registry_url})

status = gapi.repos.create_deployment_status(deployment_id=deploy.id,
                                             environment=environment,
                                             log_url=registry_url,
                                             state='success')
