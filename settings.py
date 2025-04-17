auth_file = 'authentication.yaml'
chroma_data_dir = 'chroma_db_data'

CHUNK_SIZE = 4096
CHUNK_OVERLAP = 128

dataset_dir = 'dataset'
MARSIYA_DATASET_DIR = f'{dataset_dir}/marsiya-all'
MAX_CONCURRENT_REQUESTS = 5

SUPPORTED_LLM_JUDGE_MODELS = [
    ("openai/gpt-4o-mini", True),
    ("openai/gpt-4.1-mini", False),
    ("openai/o3-mini", False),
    ("anthropic/claude-3-7-sonnet", True),
    ("anthropic/claude-3-5-haiku", False),
]