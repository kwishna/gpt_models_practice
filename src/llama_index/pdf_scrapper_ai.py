from pathlib import Path
from llama_index import download_loader

DeepDoctectionReader = download_loader("DeepDoctectionReader")

loader = DeepDoctectionReader()
documents = loader.load_data(file=Path('./data/sample_file.pdf'))

for _doc in documents:
    _text: str = _doc.text
    for __ in _text.split('\n'):
        print(__)
