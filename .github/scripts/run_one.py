from pathlib import Path
import simple_parsing, logging, os
from dataclasses import dataclass

from nb_helpers.run import run_one as _run_one

# logger.logger.setLevel(logging.DEBUG)

import weave

@weave.op
def run_one(
    fname: str,
    pip_install: bool = False,
    create_github_issue: bool = False,
    repo: str = None,
    owner: str = None,
) -> dict:
    "Run one nb and store the tb if it fails"
    run_status, runtime, tb =  _run_one(        
        fname=Path(fname),
        pip_install=pip_install,
        github_issue=create_github_issue,
        repo=repo,
        owner=owner,
        )
    return {"status": run_status == "ok", "runtime": runtime, "traceback": tb}

@dataclass
class Args:
    nb_name: str = "Chapter00.ipynb"
    project_name: str = "edu"
    entity: str = "examples_repo_test"
    pip_install: bool = True
    create_github_issue: bool = False
    issue_repo: str = None
    issue_owner: str = None

if __name__ == "__main__":
    args = simple_parsing.parse(Args, config_path="./test_config.yml")
    # let's set some env variables:
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_ENTITY"] = args.entity
    os.environ["WANDB_NAME"] = args.nb_name.split("/")[-1]

    weave.init(f"{args.entity}/{args.project_name}")

    (run_status, runtime, tb) = run_one(
        fname=args.nb_name,
        pip_install=args.pip_install,
        create_github_issue=args.create_github_issue,
        repo=args.issue_repo,
        owner=args.issue_owner,
        )