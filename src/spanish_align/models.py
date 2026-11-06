"""Data models for the pipeline: dialogue line and chunk with metadata."""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class Line:
    speaker: str
    text_es: str
    start: float
    end: float
    text_en: str | None = None
    translation_source: str = "dub"   # "dub" | "literal"


@dataclass
class Chunk:
    clip_id: str
    ig_url: str
    ig_account: str | None
    episode_id: str | None = None
    season: int | None = None
    episode: int | None = None
    episode_title: str | None = None
    match_confidence: float = 0.0
    match_method: str = "none"
    reviewed: bool = False
    chunk_id: str | None = None          # matched DB chunk (set once confirmed)
    runner_ups: list[dict] = field(default_factory=list)   # other candidates
    speakers: list[str] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_episode_id(episode_id: str) -> tuple[int, int]:
    """'S05E10' -> (5, 10)"""
    digits = episode_id.upper().replace("S", "").replace("E", "")
    return int(digits[:2]), int(digits[2:])