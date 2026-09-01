"""Owner-only E1 semantic label generation; training never imports this script."""

import argparse
import json
from pathlib import Path
import time
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_mappo.semantic_label_protocol import (
    FormalLabelSession,
    build_blind_review_pack,
    build_pilot_review_pack,
    build_semantic_prompt,
    generate_semantic_attempts,
    require_deepseek_api_key,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--model", choices=("deepseek-v4-flash", "deepseek-v4-pro"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace) -> dict:
    attempts = generate_semantic_attempts(arguments.mode)
    expected = 60 if arguments.mode == "pilot" else 800
    if len(attempts) != expected:
        raise RuntimeError("Frozen E1 scenario quota was not generated exactly.")
    arguments.output.mkdir(parents=True, exist_ok=False)
    attempts_path = arguments.output / "attempts.jsonl"
    attempts_path.write_text(
        "".join(json.dumps({
            "scenario_id": item.scenario_id, "content_hash": item.content_hash,
            "stratum": item.stratum, "semantic_view": item.semantic_view,
            "vector": item.vector,
        }, ensure_ascii=False, sort_keys=True, default=lambda value: value.item())
            + "\n" for item in attempts),
        encoding="utf-8",
    )
    if arguments.prepare_only:
        return {"mode": arguments.mode, "attempts": len(attempts), "prepared": True,
                "output": str(arguments.output), "network_calls": 0}
    key = require_deepseek_api_key()
    session = FormalLabelSession(arguments.output, arguments.model, mode=arguments.mode)
    for attempt in attempts:
        response = _request_label(key, arguments.model, build_semantic_prompt(attempt.semantic_view))
        session.consume_response(attempt, response)
    records = [json.loads(line) for line in (arguments.output / "records.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    review_pack, review_key = (
        build_pilot_review_pack(records) if arguments.mode == "pilot"
        else (build_blind_review_pack(records), [])
    )
    (arguments.output / "review_pack.json").write_text(
        json.dumps(review_pack, indent=2, sort_keys=True), encoding="utf-8"
    )
    if review_key:
        (arguments.output / "review_key_owner_only.json").write_text(
            json.dumps(review_key, indent=2, sort_keys=True), encoding="utf-8"
        )
    return {"mode": arguments.mode, "attempts": len(attempts), "prepared": False,
            "output": str(arguments.output), "network_calls": len(attempts)}


def _request_label(api_key: str, model: str, prompt) -> dict:
    body = json.dumps({
        "model": model, "stream": False, "thinking": {"type": "disabled"},
        "temperature": 0.0, "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "messages": [{"role": "system", "content": prompt.system_text},
                     {"role": "user", "content": prompt.user_text}],
    }, separators=(",", ":")).encode("utf-8")
    request = Request("https://api.deepseek.com/chat/completions", data=body,
                      headers={"Authorization": "Bearer " + api_key,
                               "Content-Type": "application/json"}, method="POST")
    last = None
    for attempt_number in range(3):
        try:
            with urlopen(request, timeout=120) as response:
                return {"status": response.status, "headers": dict(response.headers.items()),
                        "body": response.read().decode("utf-8", errors="replace")}
        except HTTPError as error:
            last = {"status": error.code, "headers": dict(error.headers.items()),
                    "body": error.read().decode("utf-8", errors="replace")}
            if error.code != 429 and error.code < 500:
                return last
        except URLError:
            last = {"status": 599, "headers": {}, "body": ""}
        if attempt_number < 2:
            time.sleep((5, 15)[attempt_number])
    return last or {"status": 599, "headers": {}, "body": ""}


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(_arguments(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
