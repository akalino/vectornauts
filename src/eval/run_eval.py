import community as community_louvain
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pinecone
import time
import torch

from collections import defaultdict
from pinecone.core.exceptions import PineconeException
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


def bng(_embed, _index, _k, _thresh):
    bsz = 100
    ids = [j for j in range(0, _embed.shape[0])]
    batched_ids = [ids[i:i+bsz] for i in range(0, _embed.shape[0], bsz)]
    batched_embs = [_embed[i:i+bsz] for i in range(0, _embed.shape[0], bsz)]
    batched_vecs = zip(batched_ids, batched_embs)
    _nn_tracker = defaultdict(set)
    print('======== Running {} batches ========'.format(len(batched_embs)))
    cnt = 0
    for batch in tqdm(batched_vecs):
        cnt += 1
        if cnt == 10:
           print(cnt)
        #    break
        else:
            ids = batch[0]
            vectors = batch[1]
            vectors = [x.tolist() for x in vectors]
            try:
                nns = _index.query(top_k=_k,
                                   queries=vectors)
            except PineconeException as e:
                print('Got error {}, sleeping and re-trying'.format(e))
                time.sleep(30)
                nns = _index.query(top_k=_k,
                                   queries=vectors)
            for i, ele in enumerate(nns['results']):
                _nn_tracker[str(ids[i])].update([res['id'] for res in ele['matches']
                                                 if ((res['score'] > _thresh)
                                                     and (str(res['id']) != str(ids[i])))])

    return _nn_tracker


def dfs(_in_id, _cluster_id, _clusters, _nn):
    """
    DFS clustering search.

    :param _in_id:
    :param _cluster_id:
    :param _clusters:
    :param _nn:
    :return:
    """
    stack = [_in_id]
    while stack:
        node = stack.pop()
        if node not in _clusters:
            _clusters[node] = _cluster_id
            stack.extend(_nn[node] - set(_clusters.keys()))


def run_dfs(_nns):
    """
    DFS clustering runner - not working due to key expansion.

    :param _nns:
    :return:
    """
    _clusters = {}
    cluster_id = 0
    for idx in _nns.keys():
        if idx not in _clusters:
            dfs(idx, cluster_id, _clusters, _nns)
            cluster_id += 1
    return _clusters


def dep_dfs(neighbours):
    """
    Deepayan DFS clustering code.

    :param neighbours:
    :return:
    """
    neighbors = defaultdict(set, neighbours)
    clusters = {}
    cluster_id = 0

    def dfs(node, cluster_id):
        stack = [node]
        while stack:
            node = stack.pop()
            if node not in clusters:
                clusters[node] = cluster_id
                stack.extend(neighbours[node] - set(clusters.keys()))

    for index in neighbours.keys():
        if index not in clusters:
            dfs(index, cluster_id)
            cluster_id += 1
    return clusters


def convert_to_adj(_nn_dict):
    """
    Converts Pinecone NN dict to local adjacency matrix.

    :param _nn_dict:
    :return:
    """
    pass


def louvain_clustering(_adj_mat):
    """
    Runs Louvain clustering.

    :param _adj_mat:
    :return:
    """
    pass


if __name__ == "__main__":
    sent_mod_name = 'gem'
    md_path = '../../data/nytfb/nytfb_metadata.json'

    PC_ENV = 'asia-northeast1-gcp'
    PC_API_KEY = ''
    PC_IDX_NAME = 'nytfb-{}-embeddings'.format(sent_mod_name)
    pinecone.init(api_key=PC_API_KEY, environment=PC_ENV)
    index = pinecone.GRPCIndex(PC_IDX_NAME)

    vecs = load_local_space(sent_mod_name)
    fil_vecs, meta = filter_local(vecs, md_path)

    try:
        with open('neigh_{}.pkl'.format(sent_mod_name), 'rb') as f:
            nn_dict = pickle.load(f)
            print('Loaded local neighbor index')
    except FileNotFoundError:
        print('No local index found, running Pinecone queries')
        nn_dict = bng(fil_vecs, index, 20, 0.7)
        with open('neigh_{}.pkl'.format(sent_mod_name), 'wb') as f:
            pickle.dump(dict(nn_dict), f)

    g = nx.DiGraph(nn_dict).to_undirected()
    print('Graph loaded, running community detection...')
    partition = community_louvain.best_partition(g)
    print('...done communities, plotting results.')
    pos = nx.spring_layout(g)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(g, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.5)
    plt.show()
    #gv, labs = filter_local(vecs, md_path)
    # tsne_embeddings(gv, sent_mod_name, labs)
    # load_from_pinecone('gem', True)
