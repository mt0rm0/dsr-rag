"""File I/O and text chunking utilities."""

import hashlib
import os
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
    """Split *text* into chunks of at most *max_words* words.

    Simple line-based splitter with no context awareness. Useful as a
    baseline for comparing against the smarter :func:`split_text`.

    Args:
        text:      Input text.
        max_words: Maximum number of words per chunk.

    Returns:
        List of non-empty text chunks.
    """
    lines = [line for line in text.splitlines(True) if line.strip()]
    chunks: List[str] = []
    chunk = ''
    for line in lines:
        if len(chunk.split()) + len(line.split()) <= max_words:
            chunk += line
        else:
            if chunk:
                chunks.append(chunk)
            chunk = line
    if chunk:
        chunks.append(chunk)
    return chunks


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
