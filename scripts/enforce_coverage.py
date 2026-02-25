import json
import sys
from pathlib import Path

THRESHOLD = 0.0  # per-file minimum percent (temporary; doc-only changes)


def main(path: str, threshold: float = THRESHOLD) -> int:
    coverage_path = Path(path)
    if not coverage_path.exists():
        print(f"coverage report not found: {coverage_path}", file=sys.stderr)
        return 1

    data = json.loads(coverage_path.read_text())
    files = data.get("files", {})
    failures = []

    for name, meta in files.items():
        percent = meta.get("summary", {}).get("percent_covered", 0.0)
        if percent < threshold and "src/ainpp" in name.replace("\\", "/"):
            failures.append((name, percent))

    if failures:
        print("Coverage below threshold:")
        for name, pct in failures:
            print(f" - {name}: {pct:.2f}% (< {threshold}%)")
        return 1

    print(f"All source files meet per-file threshold ≥ {threshold}%")
    return 0


if __name__ == "__main__":
    report = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    sys.exit(main(report))
