import itertools
import json
import torch
import pinecone

from tqdm import tqdm
#from langchain.vectorstores import Pinecone


def load_vectors(_model_name):
    """
    Loads Torch embeddings from various sentence embedding models.

    :param _model_name: Name of sentence embedding method.
    :return: Torch tensor of sentence embeddings.
    """
    _vecs = torch.load('../data/nytfb/embeddings/nytfb_{}_space.pt'.format(
        _model_name))
    return _vecs


def load_metadata(_md_path):
    """
    Loads associated dataset metadata.

    :param _md_path: Path to metadata file.
    :return:
    """
    with open(_md_path, 'r') as f:
        _meta = json.load(f)
    return _meta


def append_metadata(_sent_id, _text, _rel_idx,
                    _subjects, _objects):
    """
    Creates relevant metadata for the vectors.

    :param _sent_id: Numerical sentence identifier.
    :param _text: Text of the sentence.
    :param _relation: The sits of annotated relations present in the sentence.
    :param _rel_idx: The unique identifier for the index.
    :param _subjects: The list of subjects in the text.
    :param _subjects: The list of objects in the text.
    :return: Metadata blob.
    """
    _metadata = {
         'sentence-id': str(_sent_id),
         'text': _text,
         'rel_idx': _rel_idx,
         'subjects': _subjects,
         'objects': _objects
     }
    return _metadata


def append_index(_index, _id, _vec, _mdata):
    """
    Adds vector and metadata to the specified index.

    :param _index: Pinecone index.
    :param _vec: The embedding to be added.
    :param _mdata: Associated metadata to the embedding.
    :return: None, adds embedding to Pinecone index.
    """
    doc = {"id": _id,
            "values": _vec,
            "metadata": _mdata}
    # _index.upsert(doc)
    return doc


def chunks(_iter_list, batch_size=100):
    it = iter(_iter_list)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def run(_sent_emb_mods, _upsert):
    PC_ENV = 'asia-northeast1-gcp'
    pinecone.init(api_key=PC_API_KEY, environment=PC_ENV)
    meta = load_metadata('../data/nytfb/nytfb_metadata.json')
    for em in _sent_emb_mods:
        vecs = load_vectors(em)
        print(vecs.shape)
        PC_IDX_NAME = 'nytfb-{}-embeddings'.format(em)
        print(PC_IDX_NAME)
        index = pinecone.GRPCIndex(PC_IDX_NAME)
        print(index.describe_index_stats())
        all_data = []
        for v in tqdm(range(vecs.shape[0])):
            try:
                cmd = append_metadata(str(v),
                                      meta[str(v)]['sentence'],
                                      meta[str(v)]['rel_ids'],
                                      meta[str(v)]['subjects'],
                                      meta[str(v)]['objects'])
            except KeyError:  # OpenAI API errors
                cmd = append_metadata(str(v),
                                      '',
                                      [],
                                      [],
                                      [],
                                      [])
            all_data.append(append_index(index, str(v), vecs[v].tolist(), cmd))
        if _upsert:
            for up in chunks(all_data, 100):
                index.upsert(up)
            print('Completed building index for {}'.format(em))
            print(index.describe_index_stats())


if __name__ == "__main__":
    avail_mods = ['laser']
    run(avail_mods, True)
