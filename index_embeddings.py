import torch
import pinecone

from tqdm import tqdm
from langchain.vectorstores import Pinecone


def load_vectors(_model_name):
    """
    Loads Torch embeddings from various sentence embedding models.

    :param _model_name: Name of sentence embedding method.
    :return: Torch tensor of sentence embeddings.
    """
    _vecs = torch.load('../data/nytfb_{}_space.pt'.format(_model_name))
    return _vecs


def load_metadata(_md_path):
    """
    Loads associated dataset metadata.

    :param _md_path: Path to metadata file.
    :return:
    """
    _meta = {}
    return _meta


def append_metadata(_sent_id, _text, _relation):
    """
    Creates relevant metadata for the vectors.

    :param _sent_id: Numerical sentence identifier.
    :param _text: Text of the sentence.
    :param _relation: The annotated relation present in the sentence.
    :return: Metadata blob.
    """
    _metadata = {
         'sentence-id': str(_sent_id),
         'text': _text,
         'relation': _relation
     }
    return _metadata


def append_index(_index, _vec, _mdata):
    """
    Adds vector and metadata to the specified index.

    :param _index: Pinecone index.
    :param _vec: The embedding to be added.
    :param _mdata: Associated metadata to the embedding.
    :return: None, adds embedding to Pinecone index.
    """
    _index.upsert(vectors=zip(_vec, _mdata))


def run(_sent_emb_mod):
    vecs = load_vectors(_sent_emb_mod)
    index = pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
    meta = load_metadata('../data/nytfb-metadata')
    for v in tqdm(range(vecs.shape[0])):
        cmd = append_metadata(meta[v][0], meta[v][1], meta[v][2])
        append_index(index, vecs[v], cmd)
    print('Completed building index for {}'.format(_sent_emb_mod))


if __name__ == "__main__":
    run('gem')
