"""Entrypoints for the fixed manuscript protocol."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='GADRA manuscript-aligned mechanism simulator')
    parser.add_argument('command',choices=('prepare','evaluate','config'))
    args,remaining=parser.parse_known_args()
    if args.command=='config':
        from .profile import validate
        source=Path(remaining[0]) if remaining else Path(__file__).resolve().parents[1]/'configs/scenesplat_scannet.json'
        print(json.dumps(validate(json.loads(source.read_text())),indent=2));return
    sys.argv=[sys.argv[0]]+remaining
    if args.command=='prepare':
        from .prepare import main as prepare
        prepare()
    else:
        from .study import main as evaluate
        evaluate()

if __name__=='__main__':main()
