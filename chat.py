from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser

DEFAULT_URL = "http://localhost:8000"
QUERY_ENDPOINT = "/query"


def _post_query(base_url: str, question: str) -> dict:
    url = base_url.rstrip("/") + QUERY_ENDPOINT
    payload = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _print_separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def _format_response(data: dict) -> None:
    answer = data.get("answer", "")
    metadata = data.get("metadata", {})

    print()
    _print_separator("=")
    print(answer)
    _print_separator("=")

    # Data summary
    data_used = metadata.get("data_used", {})
    if data_used:
        period = data_used.get("period", "")
        entities = data_used.get("entities", {})
        sample = data_used.get("sample_size", 0)
        parts = []
        if period:
            parts.append(f"Period: {period}")
        if sample:
            parts.append(f"Sample: {sample}")
        for key, count in entities.items():
            if count:
                parts.append(f"{key}: {count}")
        if parts:
            print(f"  [{' | '.join(parts)}]")

    # Confidence
    limits = metadata.get("limitations_confidence", {})
    confidence = limits.get("confidence")
    if confidence is not None:
        pct = round(confidence * 100)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"  Confidence: [{bar}] {pct}%")

    # Examples
    examples = metadata.get("examples", [])
    if examples:
        print()
        print("  Examples:")
        for i, ex in enumerate(examples, 1):
            pid = ex.get("procurement_id", "?")
            etype = ex.get("entity_type", "")
            supplier = ex.get("supplier_name") or ""
            amount = ex.get("contract_sum")
            date = ex.get("sign_date") or ""
            line = f"    {i}. [{etype}] #{pid}"
            if supplier:
                line += f" — {supplier}"
            if amount:
                line += f", {amount:,.2f} KZT"
            if date:
                line += f" ({date})"
            print(line)

    # Links
    links = metadata.get("links", [])
    if links:
        print()
        print("  Links:")
        for link in links:
            print(f"    {link}")

    _print_separator("-")
    print()


def _run(base_url: str) -> None:
    print(f"Procurement AI Chat  [{base_url}]")
    print("Type your question and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue

        try:
            data = _post_query(base_url, question)
            _format_response(data)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"\n  Error {exc.code}: {body}\n")
        except urllib.error.URLError as exc:
            print(f"\n  Connection failed: {exc.reason}")
            print(f"  Is the server running at {base_url}?\n")
        except Exception as exc:
            print(f"\n  Unexpected error: {exc}\n")


def main() -> int:
    parser = ArgumentParser(description="Interactive chat with Procurement AI Agent")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()
    _run(args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
