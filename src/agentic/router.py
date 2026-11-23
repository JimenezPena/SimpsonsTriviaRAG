"""Router: decides how a free-form question is answered.

    corpus    answerable from script fragments            -> the existing RAG
    metadata  about the catalog (counts, season/number)   -> a closed tool, answer from a template
    general   general Simpsons knowledge, not in a script -> (not acted on yet)
    offtopic  nothing to do with The Simpsons             -> (not acted on yet)

Only `metadata` changes behaviour so far; every other route, and any failure
(API error, unparsable output, unknown tool, bad arguments), falls back to the
current RAG. The router can therefore only ever help.

Cost control: a cheap keyword gate runs first; questions with no season /
episode / counting words go straight to the RAG without an extra LLM call.
"""
import json
import re

from agentic import tools

ROUTES = ("corpus", "metadata", "general", "offtopic")
ROUTER_PROMPT_VERSION = 1

_GATE = re.compile(
    r"temporad|season|episod|cap[ií]tulo|\bchapter|cu[aá]nt[oa]s|how many|"
    r"\bn[uú]mero\b|\bnumber\b", re.I)


def looks_like_metadata(question):
    """Cheap pre-filter: does the question mention anything the catalog covers?"""
    return bool(_GATE.search(question or ""))


def _tool_lines():
    out = []
    for name, spec in tools.SPECS.items():
        args = ", ".join(f"{a}: {k}" for a, k in spec["args"].items())
        out.append(f"- {name}({args}): {spec['desc']}")
    return "\n".join(out)


ROUTER_SYSTEM = f"""You route questions for a Q&A system about The Simpsons. The system \
searches script fragments of the episodes it has indexed. Choose ONE route:

- "corpus": about what happens, is said or shown in the episodes (scenes, lines, \
characters' actions, plots). Includes "in which episode does X happen".
- "metadata": about the episode CATALOG itself: how many seasons/episodes, the title of \
episode N of season M, or the season/number of an episode given its title.
- "general": general knowledge about the show that no script fragment would contain \
(colours, creators, voice actors, the show's history).
- "offtopic": nothing to do with The Simpsons.

For "metadata" also choose one tool and its arguments:
{_tool_lines()}

Reply with ONLY a JSON object: {{"route": "...", "tool": null or a tool name, "args": {{}}}}.
Use the numbers or the title exactly as written in the question; do not translate titles. \
If a metadata question fits no tool, use "corpus".

Examples:
Q: How many episodes does season 4 have?
{{"route": "metadata", "tool": "count_episodes", "args": {{"season": 4}}}}
Q: ¿Qué temporada es "Lisa's Sax"?
{{"route": "metadata", "tool": "find_episode", "args": {{"title": "Lisa's Sax"}}}}
Q: What does Homer say to Marge about the bowling balls?
{{"route": "corpus", "tool": null, "args": {{}}}}
Q: ¿Quién pone la voz a Bart?
{{"route": "general", "tool": null, "args": {{}}}}
Q: Who won the football world cup?
{{"route": "offtopic", "tool": null, "args": {{}}}}"""


def parse_route(raw):
    """-> {"route", "tool", "args"} or None. Tolerates code fences and text around
    the JSON object."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("route") not in ROUTES:
        return None
    args = obj.get("args")
    return {"route": obj["route"], "tool": obj.get("tool") or None,
            "args": args if isinstance(args, dict) else {}}


def decide(question, ask, use_gate=True):
    """One routing decision. `ask(prompt, system=..., temperature=...)` is the LLM.
    -> {"route", "tool", "args", "raw", "gated_out"}; route None when the model
    could not be reached or answered garbage (the caller falls back to the RAG)."""
    if use_gate and not looks_like_metadata(question):
        return {"route": "corpus", "tool": None, "args": {}, "raw": None, "gated_out": True}
    try:
        raw = ask(f"Q: {question}", system=ROUTER_SYSTEM, temperature=0.0)
    except RuntimeError as e:
        print(f"[router] LLM error, falling back to RAG: {str(e)[:160]}", flush=True)
        return {"route": None, "tool": None, "args": {}, "raw": None, "gated_out": False}
    parsed = parse_route(raw)
    if parsed is None:
        return {"route": None, "tool": None, "args": {}, "raw": raw, "gated_out": False}
    return {**parsed, "raw": raw, "gated_out": False}


def answer_if_metadata(question, lang, ask, catalog):
    """-> {"answer", "sources": [], "route": {...}} when the question is a valid
    metadata question the tools can answer; None otherwise (use the RAG)."""
    d = decide(question, ask)
    if d["route"] != "metadata" or not d["tool"]:
        return None
    try:
        result = tools.run(d["tool"], d["args"], catalog)
    except ValueError as e:                       # unknown tool, bad or missing arguments
        print(f"[router] rejected tool call {d['tool']} {d['args']}: {e}", flush=True)
        return None
    return {"answer": tools.render(result, lang), "sources": [],
            "route": {"route": "metadata", "tool": d["tool"], "args": d["args"],
                      "ok": result["ok"]}}
