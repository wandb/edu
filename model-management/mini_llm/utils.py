import json, argparse
from ast import literal_eval
from pathlib import Path

def str2bool(v):
    "Fix Argparse to process bools"
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(config):
    print("Running with the following config")
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v) if type(v) is not bool else str2bool, 
                            default=v, 
                            help=f"Default: {v}")
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(v)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = v
        setattr(config, k, attempt)
        print(f"--{k}:{v}")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data