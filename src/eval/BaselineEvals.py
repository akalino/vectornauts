import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import time
import torch
import os
import sys
import cuml

from cuml.manifold import TSNE, UMAP
#from gudhi.clustering.tomato import Tomato
#from gtda.homology import VietorisRipsPersistence
#from gtda.plotting import plot_diagram
from sklearn.cluster import KMeans, Birch
#from cuml import KMeans
from cuml import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.utils import shuffle
from tqdm import tqdm
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score


from Classifier import MLP
# from .convex_cluster import estimate_cluster_hulls

#import tests.clusterability as clus
#from .tests.hopkins import hopkins


#def detect_persistent_clusters(_vecs):
#    t = Tomato(graph_type='radius', density_type='logKDE', k=10, r=0.1, n_jobs=-1)
#    t.fit(_vecs)
#    t.plot_diagram()
#    print('Found {} initial clusters'.format(t.n_clusters_))


#def compute_homologies(_vecs):
#    print(_vecs.shape)
#    _vecs = _vecs.reshape(1, *_vecs.shape)
#    print(_vecs.shape)
#    homology_dimensions = [0, 1, 2]
#    _vecs = _vecs.cpu().detach().numpy()
#    persistence = VietorisRipsPersistence(
#        metric="euclidean",
#        homology_dimensions=homology_dimensions,
#        n_jobs=1,
#        collapse_edges=True)
#    diagrams_basic = persistence.fit_transform(_vecs)
#    plot_diagram(diagrams_basic[0])


def calculate_hopkins(_embs, _hbins):
    embs_hopkins = []
    for i in tqdm(range(10)):
        embs_hopkins.append(hopkins(_embs, _hbins))
    return embs_hopkins, np.mean(embs_hopkins), np.std(embs_hopkins)


def apply_spatial_historgram(_embs, _bins):
    """
    Applied the spatial histogram clusterability metric.
    :param _embs: A set of PCA projected embeddings.
    :return: Spatial histogram information.
    hbins = 400
    """
    print('Running PCA')
    embs_pca = PCA(n_components=2).fit_transform(_embs)  # , svd_solver='arpack')
    print('Complete PCA')
    kls_embs = clus.spaHist(embs_pca, bins=_bins, n=50)
    mu_kls = kls_embs.mean()
    std_kls = kls_embs.std()
    print('Spatial histogram:  Mu: {m}, sigma: {s}'.format(m=mu_kls, s=std_kls))
    return kls_embs, mu_kls, std_kls


def measure_clusters(_vecs, _labels):
    _n_clust = len(list(set(_labels)))
    km = KMeans(_n_clust, verbose=0, random_state=17).fit(_vecs)
    _preds = km.labels_
    _ch = calinski_harabasz_score(_vecs, _labels)
    print("Calinski-Harabasz score: {}".format(_ch))
    return _ch


def find_best_clustering(_vecs, _labels, _kgc):
    max_groups = len(list(set(_labels)))
    print('Actual number of clusters should be {}'.format(max_groups))
    n_clusters = [max_groups]
    _prep = [_kgc] * len(n_clusters)
    sil_scores = []
    nmi_scores = []
    for _n in n_clusters:
        clusterer = KMeans(n_clusters=_n, random_state=10)
        cluster_labels = clusterer.fit_predict(_vecs)
        silhouette_avg = silhouette_score(_vecs, cluster_labels)
        print(
            "For n_clusters =",
            _n,
            " and representation via ",
            _kgc,
            "The average silhouette_score is :",
            silhouette_avg)
        sil_scores.append(silhouette_avg)
        nmi_scores.append(normalized_mutual_info_score(_labels,
                                                       cluster_labels))
    _res = pd.DataFrame({'clusters': n_clusters,
                        'score': sil_scores,
                        'preprocessing': _prep,
                         'nmi': nmi_scores})
    return _res


def find_best_cuml(_vecs, _vmat, _labels, _kgc, _project):
    """

    :return:
    """
    max_groups = len(list(set(_labels)))
    print('Actual number of clusters should be {}'.format(max_groups))
    if max_groups == 237:
        n_clusters = [100, 150, 200, max_groups, 250, 300, 350]
    else:
        n_clusters = [5, 10, max_groups, 15, 20, 25, 30]
    _prep = [_kgc] * len(n_clusters)
    sil_scores = []
    for _n in n_clusters:
        if _project:
            pca = PCA(svd_solver='full', n_components=50, random_state=17)
            _vecs = pca.fit_transform(_vecs)
            #print('Number of reduced components: {}'.format(pca.n_components_))
            #print('Explained variances: {}'.format(pca.explained_variance_ratio_))
        clusterer = KMeans(n_clusters=_n, random_state=17,)
        cluster_labels = clusterer.fit_predict(_vecs)
        #print('Fitting complete, computing silhouette')
        # silhouette_avg = silhouette_score(_vmat, cluster_labels.to_array())
        silhouette_avg = cython_silhouette_score(_vecs, cluster_labels)
        #print(
        #    "For n_clusters =",
        #    _n,
        #    " and representation via ",
        #    _kgc,
        #    "The average silhouette_score is :",
        #    silhouette_avg)
        sil_scores.append(silhouette_avg)
    _res = pd.DataFrame({'clusters': n_clusters,
                         'score': sil_scores,
                         'preprocessing': _prep})
    return _res


def measure_projected_clusters(_vecs, _labels):
    """
    In very high-dimensional spaces, Euclidean distances tend to become inflated.
    Applying dimensional reduction helps to remove this inflation and can lead to better clusters.
    :param _vecs: Set of input vectors.
    :param _labels: Set of accompanying labels.
    :return: None, prints NMI to stdout.
    """
    pca = PCA(svd_solver='full', n_components=100)
    norm_vecs = pca.fit_transform(_vecs.cpu().detach().numpy())
    print('Number of reduced components: {}'.format(pca.n_components_))
    # print('Explained variances: {}'.format(pca.explained_variance_ratio_))
    _n_clust = len(list(set(_labels)))
    print('Fitting {} clusters'.format(_n_clust))
    km = KMeans(_n_clust, verbose=0, random_state=17, n_jobs=-1, n_init=5)
    _preds = km.fit_predict(norm_vecs)
    print(_preds[0:10])
    print(_labels[0:10])
    _nmi = normalized_mutual_info_score(_labels, _preds)
    print("Normalized mutual information score: {}".format(_nmi))


def plot(_proj, _ds_name, _rel_labels):
    sns.set(style='ticks')
    sns.set_palette(sns.color_palette("Paired"))
    plt.figure(figsize=(18, 12))
    sns.scatterplot(x="component_1", y="component_2",
                    data=_proj, hue=_rel_labels)
    plt.title("TSNE Triple2Vec of {}".format(_ds_name))
    plt.show()


def apply_proj(_wvs, _type):
    RS = 17
    _wvs = _wvs.numpy()
    cuml.set_global_output_type('numpy')
    if _type == 'tsne':
        proj2 = TSNE(n_components=2,
                     perplexity=6,
                     n_neighbors=10,
                     method='fft',
                     learning_rate=2.0,
                     angle=0.8,
                     verbose=1,
                     n_iter=500,
                     n_iter_without_progress=300,
                     early_exaggeration=25.0,
                     late_exaggeration=72,
                     exaggeration_iter=500,
                     init='random',
                     random_state=RS)
    elif _type == 'umap':
        proj2 = UMAP(n_components=2,
                     n_neighbors=16,
                     n_epochs=500,
                     min_dist=0.7,
                     spread=6.94,
                     repulsion_strength=12.69,
                     negative_sample_rate=3,
                     set_op_mix_ratio=.3,
                     random_state=17)
    else:
        print('Invalid projection type, try again')
        sys.exit()
    down2 = proj2.fit_transform(_wvs, convert_dtype=True)
    projection_df = pd.DataFrame({'component_1': down2[:, 0],
                                  'component_2': down2[:, 1]})
    return projection_df


def load_matrix(_ds_name):
    _vs = torch.load("triples_{}.pt".format(_ds_name))
    return _vs


def load_model(_mod_name):
    _model = torch.load(_mod_name)
    _w = _model['triple_embedding.weight']
    return _w


def load_rel_labels():
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dp = "/data/FB15K237/unpacked/"
    fp = wd + dp + "all_triples_idx.csv"
    _triples = pd.read_csv(fp)
    return _triples


def edge_classification(_wvs, _labs):
    data_size = len(_wvs)
    print("Total number of edges to classify: {}".format(data_size))
    train_pcts = [.8]
    _wvs = shuffle(_wvs, random_state=17)
    _labs = shuffle(_labs, random_state=17)
    test_feat = _wvs[int(data_size*.8):]
    test_lab = _labs[int(data_size*.8):]
    tracker = {}
    for p in tqdm(train_pcts):
        tracker[p] = {}
        features = _wvs[0:int(data_size*p)]
        labels = _labs[0:int(data_size*p)]
        mod = LogisticRegression(multi_class="ovr",
                                 verbose=0,
                                 max_iter=1000,
                                 warm_start=True,
                                 n_jobs=1)
        clf = mod.fit(features, labels)
        _preds = clf.predict(test_feat)
        # print("Percent labeled nodes: {}".format(p))
        score = f1_score(test_lab, _preds, average="micro")
        tracker[p]['micro-f1'] = score
        print("F1 micro score: {}".format(score))
        score = f1_score(test_lab, _preds, average="macro")
        tracker[p]['macro-f1'] = score
        print("F1 macro score: {}".format(score))
        score = f1_score(test_lab, _preds, average="weighted")
        tracker[p]['weighted-f1'] = score
        print("F1 weighted score: {}".format(score))
        #if p == .8:
        #    print(classification_report(test_lab, _preds, zero_division=1))
    return tracker


def map_labels(_input):
    _label_map = {}
    _idxes = list(set(_input))
    for j in range(len(_idxes)):
        _label_map[_idxes[j]] = j
    _reversed_map = {v: k for k, v in _label_map.items()}
    return _label_map, _reversed_map


def edge_deep(_wvs, _labs, _dup):
    data_size = len(_wvs)
    n_class = len(list(set(_labs)))
    feat_dim = _wvs[0].shape[0]
    config = {"nhid": 512,
              "optim": "adam",
              "batch_size": 256,
              "usepytorch": True}
    _wvs = np.array(_wvs)
    try:
        _labs = [int(x) for x in _labs]
    except ValueError:
        lm, rlm = map_labels(_labs)
        _labs = [int(lm[x]) for x in _labs]
    if _dup:
        forward_map, reverse_map = map_labels(_labs)
        _labs = [forward_map[x] for x in _labs]
    _labs = np.array(_labs)
    rs = 17
    print("Total number of "
          "edges to classify: {} "
          "into {} classes".format(data_size, n_class))
    train_pcts = [.8]
    _wvs = shuffle(_wvs, random_state=17)
    _labs = shuffle(_labs, random_state=17)
    test_feat = _wvs[int(data_size * .8):]
    test_lab = _labs[int(data_size * .8):]
    tracker = {}
    for p in tqdm(train_pcts):
        features = _wvs[0:int(data_size * p)]
        labels = _labs[0:int(data_size * p)]
        n_train_labs = len(list(set(_labs)))
        mod = MLP(config, inputdim=feat_dim,
                  nclasses=n_train_labs,
                  seed=rs)
        mod.fit(features, labels, validation_split=0.1)
        _preds = mod.predict(test_feat)

        print("Percent labeled nodes: {}".format(p))
        tracker[p] = {}
        score = f1_score(test_lab, _preds, average="micro")
        tracker[p]['micro-f1'] = score
        print("F1 micro score: {}".format(score))
        score = f1_score(test_lab, _preds, average="macro")
        tracker[p]['macro-f1'] = score
        print("F1 macro score: {}".format(score))
        score = f1_score(test_lab, _preds, average="weighted")
        tracker[p]['weighted-f1'] = score
        print("F1 weighted score: {}".format(score))
        #if p == .8:
        #    print(classification_report(test_lab, _preds))
    return tracker


def hopkins_runner(_vecs):
    batch = int(.01 * len(_vecs))
    print('Hopkins batch size: {}'.format(batch))
    _eh, _mu, _st = calculate_hopkins(_vecs, batch)
    print('Hopkins statistics:  Mu: {m}, sigma: {s}'.format(m=_mu, s=_st))
    return _mu, _st


def run_eval(_vecs, _labs, _t, _kgc):
    vs = _vecs
    rel_labels = _labs
    if _t == 'plot':
        d2 = apply_proj(vs.cpu().detach().numpy())
        plot(d2, ds_name, rel_labels)
    elif _t == "cluster":
        measure_projected_clusters(vs, rel_labels)
    elif _t == "homology":
        detect_persistent_clusters(vs.cpu())
        # compute_homologies(vs)
    elif _t == "classify":
        print('==== OVR Results ====')
        res_1 = edge_classification(vs.cpu(), rel_labels.cpu())
        print("==== MLP Results ====")
        res_2 = edge_deep(vs.cpu(), rel_labels)
        return res_1, res_2
    elif _t == "find_best":
        kls_embs, mu_kls, std_kls = apply_spatial_historgram(vs.cpu().detach().numpy(), 100)
        odf = find_best_cuml(vs.cpu().detach().numpy(), rel_labels.cpu().detach().numpy(), _kgc)
        mu, sig = hopkins_runner(vs.cpu().detach().numpy())
        odf['hopkins_mu'] = mu
        odf['hopkins_sigma'] = sig
        odf['spahist_mu'] = mu_kls
        odf['spahist_sigma'] = std_kls
        return odf
    elif _t == "convex":
        estimate_cluster_hulls(vs.cpu(), rel_labels.cpu(), 58)
    elif _t == "collapse":
        n = 0
        vs = vs.cpu()
        while n < 20:
            for j in range(len(rel_labels)):
                n += 1
                dist = (vs[j] - vs[j + 1]).pow(2).sum(0).sqrt()
                # print(dist)
                # print(vs[j])
                # print(rel_labels[j])
                print("Distance between label {} and label {} is {}".format(rel_labels[j],
                                                                            rel_labels[j + 1],
                                                                            dist))
                time.sleep(5)

