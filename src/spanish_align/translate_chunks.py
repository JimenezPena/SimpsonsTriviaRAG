"""OFFLINE TASK: literal EN -> ES translation of the exported chunks.

Lines are merged ACROSS chunks into large batches (~BATCH_LINES per API
call): a chunk boundary does not force a new request, only the buffer
filling up does. Results are distributed back to their (chunk, index);
a chunk record is written to chunks_es.jsonl only when every one of its
lines has its translation, which keeps the run resumable by chunk_id.

    PYTHONPATH=src python -m spanish_align.translate_chunks
    PYTHONPATH=src python -m spanish_align.translate_chunks --only S01E01
    PYTHONPATH=src python -m spanish_align.translate_chunks --only S01

Resumable: chunk_ids already present in chunks_es.jsonl are skipped.
"""

import argparse
import json
import time

from .config import CHUNKS_EN, CHUNKS_ES
from .google_client import translate_literal

BATCH_LINES = 120  # lines per API call (input ~3k tokens, output ~2-3k)


def load_done() -> set[str]:
    if not CHUNKS_ES.exists():
        return set()
    return {json.loads(l)["chunk_id"]
            for l in CHUNKS_ES.read_text(encoding="utf-8").splitlines() if l.strip()}


def iter_translations(pending: list[dict]):
    """Yields (chunk, line_index, text_es), merging lines from consecutive
    chunks into single API calls until BATCH_LINES is reached."""
    buffer: list[tuple[dict, int, str]] = []

    def flush():
        texts = [t for _, _, t in buffer]
        for (chunk, idx, _), es in zip(buffer, translate_literal(texts)):
            yield chunk, idx, es
        buffer.clear()

    for chunk in pending:
        for i, line in enumerate(chunk["lines"]):
            buffer.append((chunk, i, line["text"]))
            if len(buffer) >= BATCH_LINES:
                yield from flush()
    if buffer:
        yield from flush()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Literal EN->ES translation of DB chunks")
    p.add_argument("--only",
                   help="episode filter: 'S01E10' (one episode) or 'S01' (whole season)")
    args = p.parse_args()

    chunks = [json.loads(l) for l in
              CHUNKS_EN.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = load_done()

    pending = [c for c in chunks
               if c["chunk_id"] not in done
               and (not args.only or c["episode_id"].startswith(args.only))]

    total_lines = sum(len(c["lines"]) for c in pending)
    print(f"Chunks: {len(chunks)} | translated: {len(done)} | "
          f"pending: {len(pending)} ({total_lines} lines, "
          f"~{total_lines // BATCH_LINES} API calls)")

    out: dict[str, dict] = {}          # chunk_id -> in-progress record
    written = 0
    with CHUNKS_ES.open("a", encoding="utf-8") as f:
        for chunk, idx, es in iter_translations(pending):
            rec = out.get(chunk["chunk_id"])
            if rec is None:
                rec = {**chunk, "lines": [dict(l) for l in chunk["lines"]]}
                out[chunk["chunk_id"]] = rec
            rec["lines"][idx]["text_es"] = es

            # write as soon as every line of the chunk has its translation
            if all("text_es" in l for l in rec["lines"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if written % 50 == 0:
                    print(f"  {written}/{len(pending)} chunks done")
                del out[chunk["chunk_id"]]

    print(f"Done: {written} chunks -> {CHUNKS_ES}")


if __name__ == "__main__":
    main()