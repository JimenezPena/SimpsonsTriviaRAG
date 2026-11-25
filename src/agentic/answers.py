"""Answers of the non-RAG routes: refusal (off-topic) and general knowledge."""

LANG_NAMES = {"es": "Spanish (castellano)", "en": "English"}

OFFTOPIC_REPLY = {
    "es": "Solo puedo responder preguntas sobre Los Simpson. Prueba con algo de la serie: "
          "escenas, personajes o episodios.",
    "en": "I can only answer questions about The Simpsons. Try asking about the show: "
          "scenes, characters or episodes.",
}

GENERAL_LABEL = {
    "es": "Respuesta de conocimiento general (no procede de los guiones):",
    "en": "General-knowledge answer (not taken from the scripts):",
}

GENERAL_SYSTEM = (
    "You answer general-knowledge questions about the TV show The Simpsons "
    "(production, creators, cast, history, characters, trivia). Be brief and "
    "direct (max 4 sentences) and write in the language requested.\n"
    "Never invent episode titles, quotes, dates, seasons or numbers. If you are "
    "not sure of a fact, say that you are not sure instead of guessing. If the "
    "question asks what happens or is said in a specific scene or episode, say "
    "that you cannot confirm it from general knowledge."
)


def build_general_prompt(question, lang):
    return f"Question: {question}\n\nWrite the answer in {LANG_NAMES.get(lang, LANG_NAMES['en'])}."


def label_general(answer, lang):
    """The origin label is added by code, never left to the model."""
    return f"{GENERAL_LABEL.get(lang, GENERAL_LABEL['en'])}\n{answer.strip()}"


def offtopic_reply(lang):
    return OFFTOPIC_REPLY.get(lang, OFFTOPIC_REPLY["en"])
