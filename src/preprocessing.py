"""
preprocessing.py
----------------
Cleans the 20 Newsgroups corpus for embedding.

Design decisions (justified here so they don't litter downstream code):

1. WHAT WE STRIP
   - Email headers (From:, Subject:, Organization:, Lines:, etc.) — these are
     metadata, not content. Leaving them in would teach the embedder to cluster
     on mailing-list infrastructure rather than topic.
   - Quoted reply blocks (lines starting with ">") — quoted text duplicates
     signal from other documents and inflates apparent similarity.
   - Footers / signature blocks (heuristic: lines after "-- " or repeated dashes)
   - URLs, email addresses — low semantic signal, high noise for a 384-dim model.
   - Lines that are purely punctuation or whitespace.

2. WHAT WE KEEP
   - The raw body text after headers. Even messy prose carries semantic signal.
   - Short documents are kept if >= MIN_TOKENS words remain after cleaning;
     we don't want to silently drop genuine short posts.

3. LENGTH FILTER
   - Documents with < 20 tokens after cleaning are discarded. They are almost
     always boilerplate ("Thanks.", "Me too.", etc.) with no recoverable signal.
   - We cap at MAX_TOKENS=512 during *embedding* (model context window), but
     keep the full cleaned text in the vector store metadata for retrieval.

4. CATEGORY HANDLING
   - We keep all 20 categories and expose the label as metadata. Downstream
     clustering will discover structure independently; labels are for evaluation.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOKENS = 20      # discard documents shorter than this after cleaning
MAX_TOKENS = 512     # truncate for embedding (MiniLM context limit is 256 word-pieces,
                     # but we use word-level count as a conservative proxy)

# Header fields present in virtually every 20NG post — strip entire lines matching these
HEADER_PREFIXES = (
    "from:", "subject:", "organization:", "lines:", "message-id:",
    "nntp-posting-host:", "x-newsreader:", "references:", "date:",
    "reply-to:", "sender:", "followup-to:", "distribution:", "keywords:",
    "summary:", "expires:", "supersedes:", "path:", "newsgroups:",
    "xref:", "approved:", "content-type:", "mime-version:", "in-reply-to:",
)

# Signature / footer heuristics
SIG_PATTERNS = [
    r"^--\s*$",                   # standard email sig delimiter
    r"^={4,}",                    # ==== dividers
    r"^-{4,}",                    # ---- dividers (footers, not quoted reply markers)
]
SIG_RE = re.compile("|".join(SIG_PATTERNS), re.MULTILINE)

# Quote lines — ">" at start of line (with optional whitespace/initials)
QUOTE_RE = re.compile(r"^\s{0,4}>.*$", re.MULTILINE)

# URLs and emails — replace with nothing; they carry structure but not semantics
URL_RE = re.compile(r"http\S+|www\.\S+|\S+@\S+\.\S+")

# Runs of non-alphanumeric characters (e.g. "****", "----" inline)
NOISE_RE = re.compile(r"[^a-zA-Z0-9\s.,!?;:()\'\"-]{3,}")

# Multiple blank lines → single blank line
BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class CleanedDocument:
    doc_id: str
    text: str           # cleaned, truncation-ready text
    full_text: str      # cleaned but NOT truncated — stored in vector DB metadata
    category: str
    original_length: int   # word count before cleaning
    cleaned_length: int    # word count after cleaning
    kept: bool             # False if discarded by length filter


def _strip_headers(raw: str) -> str:
    """
    Remove the header block that precedes the message body.
    Headers are separated from body by the first blank line.
    If no blank line found, we attempt prefix-based stripping.
    """
    # Standard: blank line separates header block from body
    if "\n\n" in raw:
        _, _, body = raw.partition("\n\n")
        return body

    # Fallback: strip any line that starts with a known header prefix
    lines = raw.splitlines()
    body_lines = [
        line for line in lines
        if not line.lower().startswith(HEADER_PREFIXES)
    ]
    return "\n".join(body_lines)


def _strip_quotes(text: str) -> str:
    """Remove quoted reply lines ("> ...") — they duplicate other documents."""
    return QUOTE_RE.sub("", text)


def _strip_signature(text: str) -> str:
    """
    Remove everything from the first signature delimiter onward.
    Signatures are boilerplate and actively mislead topic clustering.
    """
    match = SIG_RE.search(text)
    if match:
        text = text[: match.start()]
    return text


def _clean_text(text: str) -> str:
    """Apply URL, noise, and whitespace cleaning."""
    text = URL_RE.sub(" ", text)
    text = NOISE_RE.sub(" ", text)
    text = BLANK_RE.sub("\n\n", text)
    # Collapse runs of spaces but preserve line breaks for readability
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(lines).strip()


def clean_document(raw_text: str, doc_id: str, category: str) -> CleanedDocument:
    """
    Full cleaning pipeline for a single document.
    Returns a CleanedDocument regardless of whether it passes the length filter;
    callers check `.kept` to decide whether to ingest.
    """
    original_word_count = len(raw_text.split())

    # Pipeline: headers → quotes → signatures → noise
    text = _strip_headers(raw_text)
    text = _strip_quotes(text)
    text = _strip_signature(text)
    text = _clean_text(text)

    cleaned_word_count = len(text.split())
    kept = cleaned_word_count >= MIN_TOKENS

    # Truncated version for embedding — beyond 512 words the model attends
    # to very little of the tail anyway, so truncation is almost lossless.
    words = text.split()
    truncated = " ".join(words[:MAX_TOKENS])

    return CleanedDocument(
        doc_id=doc_id,
        text=truncated,
        full_text=text,
        category=category,
        original_length=original_word_count,
        cleaned_length=cleaned_word_count,
        kept=kept,
    )


def preprocess_corpus(raw_docs: list[dict]) -> tuple[list[CleanedDocument], dict]:
    """
    Process a list of raw documents.

    Args:
        raw_docs: list of {"id": str, "text": str, "category": str}

    Returns:
        (kept_docs, stats) where stats summarises what was discarded and why.
    """
    kept, discarded = [], []
    category_counts: dict[str, int] = {}

    for doc in raw_docs:
        cleaned = clean_document(
            raw_text=doc["text"],
            doc_id=doc["id"],
            category=doc["category"],
        )
        if cleaned.kept:
            kept.append(cleaned)
            category_counts[cleaned.category] = category_counts.get(cleaned.category, 0) + 1
        else:
            discarded.append(cleaned)

    stats = {
        "total_raw": len(raw_docs),
        "kept": len(kept),
        "discarded": len(discarded),
        "discard_rate": round(len(discarded) / max(len(raw_docs), 1), 3),
        "category_distribution": category_counts,
        "avg_cleaned_length": round(
            sum(d.cleaned_length for d in kept) / max(len(kept), 1), 1
        ),
    }

    logger.info(
        f"Preprocessing complete: {stats['kept']} kept, "
        f"{stats['discarded']} discarded ({stats['discard_rate']*100:.1f}% discard rate)"
    )
    return kept, stats
