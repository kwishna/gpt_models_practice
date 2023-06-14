from pathlib import Path
from llama_index import download_loader
from llama_index.readers.file.docs_reader import PDFReader

_PDFReader = download_loader("PDFReader")

loader: PDFReader = _PDFReader()
documents = loader.load_data(file=Path('../data.pdf'))
for _doc in documents:
    _text: str = _doc.text
    for __ in _text.split('\n'):
        print(__)
