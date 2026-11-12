from beir import util
from beir.datasets.data_loader import GenericDataLoader

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

def load_scifact(split="test", root="data"):
    path = util.download_and_unzip(URL, root)
    return GenericDataLoader(path).load(split=split)   # corpus, queries, qrels

def doc_text(doc):
    return (doc.get("title", "") + " " + doc["text"]).strip()