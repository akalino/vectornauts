import json
import pinecone
import torch

from tqdm import tqdm

from BaselineEvals import apply_proj, plot


def load_local_space(_mn):
    """
    Loads embedding space.

    :param _mn: Model name.
    :return: Torch matrix.
    """
    _emb = torch.load('../../data/nytfb/embeddings/nytfb_{}_space.pt'.
                      format(_mn))
    return _emb


def load_metadata(_md_path):
    """
    Loads associated dataset metadata.

    :param _md_path: Path to metadata file.
    :return:
    """
    with open(_md_path, 'r') as f:
        _meta = json.load(f)
    return _meta


def filter_local(_vecs, _md_path):
    _filtered = []
    _rel_lab = []
    meta = load_metadata(_md_path)
    for v in tqdm(range(_vecs.shape[0])):
        if meta[str(v)]['rel_ids'] != ["35"]:
            _filtered.append(_vecs[v])
            _rel_lab.append(meta[str(v)]['rel_ids'][0])
    _filtered = torch.stack(_filtered)
    print(_filtered.shape)
    return _filtered, _rel_lab


def load_from_pinecone(_sent_model, _filter):
    # too slow
    PC_ENV = 'asia-northeast1-gcp'
    PC_API_KEY = '766c9273-752a-48ca-bd5f-26c9c17201a8'
    PC_IDX_NAME = 'nytfb-{}-embeddings'.format(_sent_model)
    pinecone.init(api_key=PC_API_KEY, environment=PC_ENV)
    index = pinecone.GRPCIndex(PC_IDX_NAME)
    vc = index.describe_index_stats()['total_vector_count']
    nn_embs = []
    for v_idx in tqdm(range(vc)):
        if _filter:
            qr = index.query(id=str(v_idx),
                             top_k=1,
                             include_values=True,
                             filter={'rel_idx': {'$ne': '35'}})
            nn_embs.append(qr['matches'][0]['values'])
        else:
            qr = index.query(id=str(v_idx),
                             top_k=1,
                             include_values=True)
            nn_embs.append(qr['matches'][0]['values'])
    print('Queries for {} total vectors'.format(len(nn_embs)))


def tsne_embeddings(_emb, _mn, _labs):
    """
    Plots low-dimensional projection of embeddings.

    :param _emb: PyTorch embedding matrix.
    :return: None, plots.
    """
    type = 'umap'
    proj_df = apply_proj(_emb, type)
    plot(proj_df, _mn, _labs)


if __name__ == "__main__":
    sent_mod_name = 'openai'
    md_path = '../../data/nytfb/nytfb_metadata.json'
    vecs = load_local_space(sent_mod_name)
    gv, labs = filter_local(vecs, md_path)
    tsne_embeddings(gv, sent_mod_name, labs)
    # load_from_pinecone('gem', True)
