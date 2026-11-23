"""Prompt construction for the distillation dataset.

The student is trained on EXACTLY what the production free-form Q&A sends to
the teacher, so the layout below mirrors `api_ask` in src/app.py:

    Context:\n\n<plot summary?>\n\n---\n\n<chunk 1>\n\n---\n\n<chunk 2> ...
    \n\nQuestion: <q>\n\nWrite the answer in <language>.

If you change the prompt in app.py, change it here too (and regenerate the
data): a train/serve mismatch silently costs accuracy.
"""

# language code -> name used inside the prompt (same as app.py). Only es/en for
# now; to add a language also add its stopwords in build_dataset._STOP.
LANG_NAMES = {
    "es": "Spanish (castellano)",
    "en": "English",
}


def format_context(chunks, plot=None):
    """chunks: iterable of (season, episode_title, text), best match first.
    plot: optional dict as returned by trivia_gen.plot_for_question."""
    ctx = "\n\n---\n\n".join(
        f"[Season {season}, \"{title}\"]\n{text}" for season, title, text in chunks)
    if plot:
        ctx = (f"[Plot summary (Wikipedia), season {plot['season']}, "
               f"\"{plot['title']}\"]\n" + "\n\n".join(plot["paragraphs"])
               + "\n\n---\n\n" + ctx)
    return ctx


def build_qa_prompt(context, question, lang):
    return (f"Context:\n\n{context}\n\nQuestion: {question}\n\n"
            f"Write the answer in {LANG_NAMES[lang]}.")


# --- synthetic user questions -------------------------------------------------
QUESTION_PROMPT_VERSION = 3     # stored in every record; v1: no specificity rules; v2: self-contained;
#                                 v3: every fact in the question must come from the fragment

QUESTION_SYSTEM = (
    "You help build a test set for a question-answering system about The "
    "Simpsons. The system searches a database of script fragments from ALL "
    "episodes, so every question must be SELF-CONTAINED: someone who has not "
    "seen the fragment must be able to tell which scene it is about.\n"
    "You receive a script fragment and write ONE short question, phrased the "
    "way a fan would type it into a search box. Rules:\n"
    "- Anchor the question in at least one distinctive element of the scene: "
    "a secondary character's name, a specific object, place, event or "
    "memorable line. The anchor must come from the fragment.\n"
    "- Never rely on vague references such as 'the school', 'the hotel', "
    "'the kids', 'a man', 'this episode', and never ask about a generic "
    "action of a main character ('What is Marge looking for?') or depend on "
    "the fragment for context ('while talking to Lisa').\n"
    "- Skip trivial details; ask about something a fan would plausibly "
    "want to know.\n"
    "- Never mention 'the fragment', 'the text' or 'the script'.\n"
    "- Every name, object and claim in the question must appear in the "
    "fragment. Do not use outside knowledge about the show, and do not "
    "assume anything the fragment does not state (no false premises: do not "
    "ask why someone did something unless the fragment says they did it).\n"
    "- Ask about what is explicitly said or shown in the fragment, not about "
    "what is never revealed in it (reasons, motives, offscreen events).\n"
    "Return ONLY the question."
)


def build_question_prompt(text, season, title, lang, answerable=True):
    relation = ("is clearly and fully answered by" if answerable else
                "is about the same characters or episode but is NOT answered by")
    return (f"Script fragment (season {season}, episode \"{title}\"):\n\n{text}\n\n"
            f"Write a question in {LANG_NAMES[lang]} that {relation} this fragment.")


def clean_question(raw):
    """First non-empty line, without quotes or list markers; None if it does
    not look like a question."""
    for line in (raw or "").splitlines():
        line = line.strip().strip("\"'“”«»*-• ").strip()
        if line:
            return line if "?" in line and len(line) >= 8 else None
    return None
