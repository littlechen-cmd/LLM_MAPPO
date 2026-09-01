"""Aggregate an owner-run E1 CUDA smoke without inspecting performance."""

import argparse
import json

from llm_mappo.e1_smoke import aggregate_cuda_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_cuda_smoke(args.root)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
