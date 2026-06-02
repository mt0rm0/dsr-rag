from os import getenv
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

DATA_PATH = 'data'

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

VECTOR_DB_PATH = 'vector_db'
COLLECTION_NAME = 'my_docs'

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = [
    'multi-qa-mpnet-base-cos-v1',
    'all-mpnet-base-v2',
    'nomic-ai/nomic-embed-text-v1.5',
    'multi-qa-MiniLM-L6-cos-v1',
    'all-MiniLM-L6-v2',
]

# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------

LLMBASEURL = urljoin(getenv('OLLAMA_HOST', 'http://localhost:11434'), 'api')

LLM_CHOICES = [
    'llama3.2',
    'llama3.1',
    'phi3.5',
    'mistral',
    'gemma2',
    'qwen2.5',
    'qwen2.5:3b',
    'smollm2',
]

# ---------------------------------------------------------------------------
# Retrieval behaviour
# ---------------------------------------------------------------------------

KEYWORD_SEARCH = True    # Fall back to keyword search when semantic search returns nothing
FILTER_BY_KEYWORD = True  # Refine semantic results with an extracted keyword

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

CONVERSATION_LENTGH = 10  # Max interactions kept in dialogue history

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPORT_PATH = 'exports'
LOG_DIR = 'log'
LOG_FILE = 'info.log'
