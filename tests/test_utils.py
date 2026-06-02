"""Tests for ragsst.utils — chunking and file I/O."""

import pytest
from ragsst.utils import list_files, read_file, split_text, split_text_basic


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------

def test_split_text_returns_list():
    chunks = split_text('This is a sentence.\nThis is another sentence.')
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_split_text_respects_max_words():
    line = ' '.join(['word'] * 50) + '.'
    text = '\n'.join([line] * 6)
    chunks = split_text(text, max_words=100)
    for chunk in chunks:
        assert len(chunk.split()) <= 110  # small tolerance for edge lines


def test_split_text_no_empty_chunks():
    text = '\n\n\nSome text here.\n\n\nMore text here.\n\n'
    for chunk in split_text(text):
        assert chunk.strip() != ''


def test_split_text_preserves_content():
    text = 'The quick brown fox.\nJumped over the lazy dog.'
    joined = ' '.join(split_text(text))
    assert 'quick brown fox' in joined
    assert 'lazy dog' in joined


def test_split_text_single_line():
    chunks = split_text('Just one line of text.')
    assert len(chunks) == 1
    assert 'Just one line' in chunks[0]


def test_split_text_basic_nonempty():
    text = 'Paragraph one.\nParagraph two.\nParagraph three.'
    assert len(split_text_basic(text)) >= 1
    assert len(split_text(text)) >= 1


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

def test_list_files_returns_sorted(tmp_path):
    (tmp_path / 'b.txt').write_text('b')
    (tmp_path / 'a.txt').write_text('a')
    files = list_files(str(tmp_path), extensions=('.txt',))
    names = [f.split('/')[-1] for f in files]
    assert names == sorted(names)


def test_list_files_filters_by_extension(tmp_path):
    (tmp_path / 'doc.txt').write_text('text')
    (tmp_path / 'doc.pdf').write_text('pdf')
    (tmp_path / 'doc.py').write_text('python')
    txt_files = list_files(str(tmp_path), extensions=('.txt',))
    assert all(f.endswith('.txt') for f in txt_files)
    assert len(txt_files) == 1


def test_list_files_empty_dir(tmp_path):
    assert list_files(str(tmp_path)) == []


def test_list_files_walksubdirs(tmp_path):
    subdir = tmp_path / 'sub'
    subdir.mkdir()
    (subdir / 'nested.txt').write_text('nested')
    (tmp_path / 'root.txt').write_text('root')
    assert len(list_files(str(tmp_path), walksubdirs=True, extensions=('.txt',))) == 2
    assert len(list_files(str(tmp_path), walksubdirs=False, extensions=('.txt',))) == 1


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_txt(tmp_path):
    doc = tmp_path / 'test.txt'
    doc.write_text('Hello from a text file.')
    assert 'Hello from a text file.' in read_file(str(doc))


def test_read_file_empty_txt(tmp_path):
    doc = tmp_path / 'empty.txt'
    doc.write_text('')
    assert read_file(str(doc)) == ''


def test_read_file_unsupported_returns_empty(tmp_path):
    doc = tmp_path / 'test.csv'
    doc.write_text('a,b,c')
    assert read_file(str(doc)) == ''
