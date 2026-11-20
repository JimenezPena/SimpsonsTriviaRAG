"""Prompt construction for the distillation dataset.

The student is trained on EXACTLY what the production free-form Q&A sends to
the teacher, so the layout below mirrors `api_ask` in src/app.py:

    Context:\n\n<plot summary?>\n\n---\n\n<chunk 1>\n\n---\n\n<chunk 2> ...
    \n\nQuestion: <q>\n\nWrite the answer in <language>.

If you change the prompt in app.py, change it here too (and regenerate the
data): a train/serve mismatch silently costs accuracy.
"""

# language code -> name used inside the prompt. app.py only uses es/en today;
# ca/de are here for the multilingual experiments.
LANG_NAMES = {
    "es": "Spanish (castellano)",
    "en": "English",
    "ca": "Catalan",
    "de": "German",
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
QUESTION_SYSTEM = (
    "You help build a test set for a question-answering system about The "
    "Simpsons. You receive a script fragment and write ONE short question, "
    "phrased the way a fan would type it into a search box. Never mention "
    "'the fragment', 'the text' or 'the script'. Return ONLY the question."
)


def build_question_prompt(text, season, title, lang, answerable=True):
    relation = ("is clearly answered by" if answerable else
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
