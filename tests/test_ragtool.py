"""Tests for ragsst.ragtool — RAGTool core methods."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from ragsst.ragtool import RAGTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool(tmp_path):
    """RAGTool with mocked ChromaDB and a temp data directory."""
    with patch('ragsst.ragtool.chromadb.PersistentClient') as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'metadatas': []}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_client.return_value.list_collections.return_value = []

        t = RAGTool(data_path=str(tmp_path), collection_name='test_collection')
        t.collection = mock_collection
        yield t


# ---------------------------------------------------------------------------
# get_context_prompt
# ---------------------------------------------------------------------------

def test_context_prompt_contains_query(tool):
    prompt = tool.get_context_prompt('What is RAG?', 'Some context.')
    assert 'What is RAG?' in prompt


def test_context_prompt_contains_context(tool):
    context = 'RAG stands for Retrieval Augmented Generation.'
    assert context in tool.get_context_prompt('question', context)


def test_context_prompt_is_string(tool):
    assert isinstance(tool.get_context_prompt('q', 'c'), str)


# ---------------------------------------------------------------------------
# get_condenser_prompt
# ---------------------------------------------------------------------------

def test_condenser_prompt_contains_followup(tool):
    history = deque(['Q: What is Berlin?', 'A: The capital of Germany.'])
    prompt = tool.get_condenser_prompt('And Munich?', history)
    assert 'And Munich?' in prompt


def test_condenser_prompt_contains_history(tool):
    history = deque(['Q: What is Berlin?', 'A: The capital of Germany.'])
    prompt = tool.get_condenser_prompt('And Munich?', history)
    assert 'Berlin' in prompt


# ---------------------------------------------------------------------------
# get_relevant_text
# ---------------------------------------------------------------------------

def test_get_relevant_text_returns_string(tool):
    tool.collection.query.return_value = {
        'ids': [['id1']],
        'documents': [['Some relevant text.']],
        'distances': [[0.1]],
        'metadatas': [[{'source': 'doc.txt'}]],
    }
    result = tool.get_relevant_text('test query', nresults=1)
    assert isinstance(result, str)


def test_get_relevant_text_filters_below_threshold(tool):
    # Distance 0.8 → similarity 0.2, below threshold 0.5 → empty result
    tool.collection.query.return_value = {
        'ids': [['id1']],
        'documents': [['Low relevance text.']],
        'distances': [[0.8]],
        'metadatas': [[{'source': 'doc.txt'}]],
    }
    result = tool.get_relevant_text('query', nresults=1, sim_th=0.5, keyword_search=False)
    assert result == ''


def test_get_relevant_text_passes_above_threshold(tool):
    # Distance 0.1 → similarity 0.9, above threshold 0.5 → returns text
    tool.collection.query.return_value = {
        'ids': [['id1']],
        'documents': [['Highly relevant text.']],
        'distances': [[0.1]],
        'metadatas': [[{'source': 'doc.txt'}]],
    }
    result = tool.get_relevant_text('query', nresults=1, sim_th=0.5, keyword_filter=False)
    assert 'Highly relevant text.' in result


# ---------------------------------------------------------------------------
# list_collections_names
# ---------------------------------------------------------------------------

def test_list_collections_names_empty(tool):
    tool.vs_client.list_collections.return_value = []
    assert tool.list_collections_names() == []


def test_list_collections_names_returns_names(tool):
    mock_col = MagicMock()
    mock_col.name = 'my_collection'
    tool.vs_client.list_collections.return_value = [mock_col]
    assert 'my_collection' in tool.list_collections_names()
