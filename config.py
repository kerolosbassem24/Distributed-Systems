import torch

# LLM
LLM_MODEL_NAME       = "microsoft/phi-2"       # or "mistralai/Mistral-7B-Instruct-v0.2"
                                                # or any HuggingFace causal LM
LLM_MAX_NEW_TOKENS   = 256
LLM_TEMPERATURE      = 0.7
LLM_DO_SAMPLE        = True
LLM_DTYPE            = torch.float16           # fp16 for GPU memory efficiency

# GPU
NUM_GPUS             = torch.cuda.device_count() if torch.cuda.is_available() else 0
GPU_MEMORY_FRACTION  = 0.85                        # reserve 85% of each GPU

# Load Balancing
LOAD_BALANCE_STRATEGY = "load_aware"           # "round_robin" | "least_connections" | "load_aware"

# RAG / Vector DB
VECTOR_DB_PATH       = "./chroma_db"
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CHUNKS         = 3
CHUNK_SIZE           = 500
CHUNK_OVERLAP        = 50

# Fault Tolerance
HEARTBEAT_INTERVAL   = 5        # seconds
FAILURE_THRESHOLD    = 3        # missed heartbeats before marking dead
MAX_RETRIES          = 3        # task reassignment retries

# Load Test
NUM_USERS            = 100
THREAD_POOL_SIZE     = 100

# Testing Modes
MULTI_SCALE_TEST         = False
FAILURE_TEST             = False
STRATEGY_COMPARISON_TEST = False
