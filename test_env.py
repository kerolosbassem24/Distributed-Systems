import torch
from transformers import __version__ as tr_version
import chromadb
import sentence_transformers

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}:", torch.cuda.get_device_name(i))
print("Transformers Version:", tr_version)
print("ChromaDB Version:", chromadb.__version__)
print("Sentence-Transformers Version:", sentence_transformers.__version__)
print("Environment Test Passed!")
