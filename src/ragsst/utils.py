"""File I/O and text chunking utilities."""

import hashlib
import os
import re
from typing import List, Tuple

from docx import Document
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def list_files(
    path: str,
    walksubdirs: bool = True,
    extensions: str | Tuple[str, ...] = '',
) -> List[str]:
    """Return a sorted list of file paths under *path* matching *extensions*.

    Args:
        path:         Root directory to search.
        walksubdirs:  If True, recurse into subdirectories.
        extensions:   A string or tuple of strings (e.g. ``('.txt', '.pdf')``).
                      Pass an empty string to match all files.

    Returns:
        Sorted list of absolute file paths.
    """
    if walksubdirs:
        files = [
            os.path.join(root, f)
            for root, _dirs, files in os.walk(path)
            for f in files
            if f.endswith(extensions)
        ]
    else:
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(extensions)
        ]
    return sorted(files)


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_file(doc: str) -> str:
    """Extract plain text from a .txt, .pdf, or .docx file.

    Args:
        doc: Path to the file.

    Returns:
        Extracted text, or an empty string for unsupported formats.
    """
    if doc.endswith('.txt'):
        with open(doc, 'r', encoding='utf-8') as f:
            return f.read()
    if doc.endswith('.pdf'):
        reader = PdfReader(doc)
        return ''.join(page.extract_text() or '' for page in reader.pages)
    if doc.endswith('.docx'):
        document = Document(doc)
        return '\n'.join(para.text for para in document.paragraphs)
    return ''


def hash_file(filename: str, block_size: int = 128 * 64) -> str:
    """Return the SHA-1 hex digest of a file's contents.

    Used to detect whether a file has changed since it was last ingested.
    """
    h = hashlib.sha1()
    with open(filename, 'rb') as f:
        while data := f.read(block_size):
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def split_text_basic(text: str, max_words: int = 256) -> List[str]:
    """Split *text* into fixed-size chunks of at most *max_words* words.

    Tokenises the entire text at the word level, then packs words into
    chunks. This guarantees that every chunk is within *max_words* words,
    even when individual lines are longer than the limit. Whitespace
    structure is not preserved — this is a pure word-count splitter useful
    as a baseline.

    Args:
        text:      Input text.
        max_words: Maximum number of words per chunk.

    Returns:
        List of non-empty text chunks.

    Example::

        >>> chunks = split_text_basic("one two three four five", max_words=3)
        >>> chunks
        ['one two three', 'four five']
    """
    words = text.split()
    return [
        ' '.join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def split_text(text: str, max_words: int = 256, max_title_words: int = 4) -> List[str]:
    """Split *text* into context-aware chunks of at most *max_words* words.

    Short lines (≤ *max_title_words* words) that look like headings or titles
    are kept together with the paragraph that follows them rather than being
    split off on their own.

    Args:
        text:            Input text.
        max_words:       Maximum number of words per chunk.
        max_title_words: Lines with this many words or fewer are treated as
                         potential headings and kept with the next paragraph.

    Returns:
        List of non-empty text chunks.

    Example::

        >>> chunks = split_text("Introduction\\nThis is a paragraph about RAG.")
        >>> len(chunks)
        1
    """
    punctuations = ('.', '?', '!', '\u201d', '"')
    lines = [line for line in text.splitlines() if line.strip()]

    chunks: List[str] = []
    chunk: List[str] = []
    chunk_length = 0

    for line in lines:
        line_length = len(line.split())
        keep_together = (
            chunk_length + line_length <= max_words
            and (
                line_length > max_title_words
                or line.strip().endswith(punctuations)
                or all(len(s.split()) <= max_title_words for s in chunk)
            )
        )
        if keep_together:
            chunk.append(line)
            chunk_length += line_length
        else:
            if chunk:
                chunks.append('\n'.join(chunk))
            chunk = [line]
            chunk_length = line_length

    if chunk:
        chunks.append('\n'.join(chunk))

    return chunks


def split_text_sliding_window(
    text: str,
    max_words: int = 256,
    overlap: int = 32,
) -> List[str]:
    """Split *text* into overlapping fixed-size word chunks.

    Adjacent chunks share *overlap* words, so a sentence that falls near a
    chunk boundary appears in both neighbours. This reduces the chance of
    splitting a relevant passage across two chunks that are never retrieved
    together.

    Args:
        text:      Input text.
        max_words: Number of words per chunk.
        overlap:   Number of words shared between consecutive chunks.
                   Must be less than *max_words*.

    Returns:
        List of non-empty text chunks.

    Example::

        >>> chunks = split_text_sliding_window("a b c d e f", max_words=4, overlap=2)
        >>> chunks
        ['a b c d', 'c d e f']
    """
    if overlap >= max_words:
        raise ValueError('overlap must be less than max_words')

    words = text.split()
    step = max_words - overlap
    return [
        ' '.join(words[i:i + max_words])
        for i in range(0, len(words), step)
        if words[i:i + max_words]
    ]


def split_text_sentences(
    text: str,
    max_words: int = 256,
    overlap_sentences: int = 1,
) -> List[str]:
    """Split *text* into chunks that respect sentence boundaries.

    Sentences are packed into a chunk until adding the next sentence would
    exceed *max_words*. The last *overlap_sentences* sentences of each chunk
    are prepended to the next one to preserve local context across boundaries.

    Args:
        text:               Input text.
        max_words:          Maximum number of words per chunk.
        overlap_sentences:  Number of sentences from the end of each chunk
                            to repeat at the start of the next. Set to 0 to
                            disable overlap.

    Returns:
        List of non-empty text chunks.
    """
    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If a single sentence exceeds the limit, emit it as its own chunk
        if sentence_words > max_words:
            if current:
                chunks.append(' '.join(current))
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_words = sum(len(s.split()) for s in current)
            chunks.append(sentence)
            continue

        if current_words + sentence_words > max_words:
            chunks.append(' '.join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(' '.join(current))

    return chunks


def split_text_paragraphs(
    text: str,
    max_words: int = 256,
) -> List[str]:
    """Split *text* into chunks that respect paragraph boundaries.

    Paragraphs (separated by one or more blank lines) are packed together
    until adding the next paragraph would exceed *max_words*. A paragraph
    that is itself longer than *max_words* is emitted as its own chunk
    without further splitting.

    This strategy works well for structured documents (articles, books)
    where paragraphs already represent coherent units of thought.

    Args:
        text:      Input text.
        max_words: Soft maximum number of words per chunk.

    Returns:
        List of non-empty text chunks.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > max_words:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_words = 0

        current.append(para)
        current_words += para_words

    if current:
        chunks.append('\n\n'.join(current))

    return chunks
