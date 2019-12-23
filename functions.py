import pandas as pd
from tqdm.auto import tqdm
from razdel import sentenize, tokenize
import networkx as nx
from networkx.algorithms import bipartite
import gensim
from gensim.models import KeyedVectors
from pymystem3 import Mystem
from string import punctuation
import numpy as np
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import matplotlib.pyplot as plt
import matplotlib
from stop_words import get_stop_words
from scipy.optimize import linear_sum_assignment
from scipy import spatial
from munkres import make_cost_matrix
import joblib
from collections import Counter
from community.community_louvain import best_partition


punct = punctuation+'«»—…“”*–'
stop_words = get_stop_words('ru')


def read_freqs(path):
    """reads russian corpora frequencies"""
    freq_dict = {}
    with open(path, "r") as f: 
        f = f.readlines()
        for line in f:
            if line == "\n" or line == "":
                continue
            line = line.strip("\n").split("\t")
            freq_dict[line[1]] = int(line[0])
    return freq_dict


def lemmatize(word, morph):
    processed = morph.analyze(word)[0]
    try:
        lemma = processed["analysis"][0]["lex"].lower().strip()
    except:
        lemma = ""
    return lemma


def freq_sums(freq_dict, morph):
    "lemmatizes frequency dictionary and sums up word counts with the same lemmas"
    new_freq_dict = {}
    for word in tqdm(freq_dict):
        lemma = lemmatize(word, morph)
        if lemma == "":
            continue
        if lemma in new_freq_dict:
            new_freq_dict[lemma] += freq_dict[word]
        else:
            new_freq_dict[lemma] = freq_dict[word]
    return new_freq_dict


def get_embedding_dicts(text_tokens:list, model):
    """returns embedding dictionary for all words within list of sentences"""
    emb_dicts = []
    for sent in text_tokens:
        d = {}
        for word in sent:
            if word in stop_words:
                continue
            vector = np.zeros(shape=model.vector_size)
            try:
                vector = np.add(vector, model[word])
                d[word] = vector
            except KeyError:
                pass
        emb_dicts.append(d)
    return emb_dicts


def get_sent_pairs(lst):
    """creates list of item pairs from the list of items"""
    new_lst = []
    for i in range(len(lst)-1):
        new_lst.append(lst[i:i+2])
    return new_lst

def get_optimal_alignment(dict_pairs, model):
    
    """creates a weighted complete bipartite graph between the sets of content words of the two sentences and runs a bipartite graph matching algorithm known as the Hungarian method"""
    
    aligned_sents = []
    
    for pair in dict_pairs:
        words_1 = [w for w in list(pair[0].keys()) if w not in stop_words and w not in punct]
        words_2 = [w for w in list(pair[1].keys()) if w not in stop_words and w not in punct]
        nodes_1 = [i + "_&" for i in words_1]
        nodes_2 = [i + "_@" for i in words_2]
        
        edges = []
        for i,w_1 in enumerate(words_1):
            for j,w_2 in enumerate(words_2):
                weight = model.similarity(w_1, w_2)
                edge = (nodes_1[i], nodes_2[j], {"weight":weight})
                edges.append(edge)
                
        B = nx.Graph()
        B.add_nodes_from(nodes_1, bipartite=0)
        B.add_nodes_from(nodes_2, bipartite=1)
        B.add_edges_from(edges)
        
        M = biadjacency_matrix(B, row_order=nodes_1, column_order=nodes_2).todense()
        M = np.array(M)
        minimum = min(min(row) for row in M)
        M += abs(minimum) 
        M *= 100
        M = M.astype(int)
        cost_matrix = np.array(make_cost_matrix(M))
        row_index, col_index = linear_sum_assignment(cost_matrix)
        
        aligned_words = []
        for i,j in zip(row_index, col_index): 
            aligned_words.append((words_1[i],words_2[j])) 
        aligned_sents.append(aligned_words)

    return aligned_sents


def ic(word, freq_sums):
    """computes the information content of the word, based of the relative frequency of word in a large corpus"""
    try: 
        word_freq = freq_sums[word]
    except KeyError:
        word_freq = 0
    return -np.log((word_freq + 1)/(len(freq_sums) + sum(freq_sums.values())))


def sr_scores(aligned_sents, model, freq_sums):
    """compites the semantic relatedness score between sentences"""
    sr_scores = []
    for pair in aligned_sents:
        sr = 0
        for w_1, w_2 in pair:
            sem_closeness = model.similarity(w_1, w_2) * min(ic(w_1, freq_sums), ic(w_2, freq_sums))
            sr += sem_closeness 
        norm_sr = sr / len(pair)
        sr_scores.append(norm_sr)
    return sr_scores    


def get_segments(tau, sent_indexes, sr_dict):
    
    """returns list of segments containing sentence indexes given their semantic relatedness scores"""
    
    # constructing the similarity graph
    G = nx.Graph()
    G.add_nodes_from(sent_indexes)
    for pair in sr_dict.keys():
        if sr_dict[pair] > tau:
            G.add_edge(pair[0], pair[1])
            
    # get sorted subgraphs        
    subgraphs = sorted([sorted(list(i)) for i in list(nx.connected_components(G))])
    # preds = [sents[block[0]:block[-1]+1] for block in graphs]
    
    # merging standalone segments if they go one after another
    
    segments = []
    cur_list = []
    
    for i in range(len(subgraphs)):
        if len(subgraphs[i]) == 1:
            cur_list.append(subgraphs[i][0])
        else:
            segments.append(subgraphs[i])
    
    pairs = get_sent_pairs(cur_list)
    
    d = {}
    for pair in pairs:
        if pair[1]-pair[0] == 1:
            d[(pair[0], pair[1])] = 1
        else:
            d[(pair[0], pair[1])] = 0
            
    new_G = nx.Graph()
    for indx in cur_list:
        new_G.add_node(indx)
    for pair in d.keys():
        if d[pair] == 1:
            new_G.add_edge(pair[0], pair[1])
            
    new_graphs = sorted([sorted(list(i)) for i in list(nx.connected_components(new_G))])
    for i in new_graphs:
        segments.append(i)
        
    return sorted(segments)


def segmentize_bipartite_subgraphs(path_to_text, model, tau):
    
#    freq_dict = read_freqs("1grams-3.txt")
#    freqs = freq_sums(freq_dict, morph=Mystem())
    freqs = joblib.load("freqs.pkl")
    
    with open(path_to_text, "r") as file:
        file = file.readlines()
    paragraphs = [line for line in file if not line.startswith("#") and line != "\n"]
    
    real_paragraphs = []
    sents = []
    for par in paragraphs:
        sent_list = [_.text for _ in sentenize(par)]
        real_paragraphs.append(sent_list)
        sents.extend(sent_list)
        
    # соберем и лемматизируем токены
    tokens = []
    m = Mystem()
    print("Lemmatizing sentences")
    for sent in tqdm(sents):
        t = [_.text for _ in tokenize(sent)]
        lemmas = []
        for token in t:
            lemma = lemmatize(token, morph=m)
            if lemma != "":
                lemmas.append(lemma)
        tokens.append(lemmas)
    
    emb_dicts = get_embedding_dicts(tokens, model=model)
    dict_pairs = get_sent_pairs(emb_dicts)
    aligned_sents = get_optimal_alignment(dict_pairs, model=model)
    sr = sr_scores(aligned_sents, model=model, freq_sums=freqs)
    sents_indexes = [i for i in range(len(sents))]
    sr_dict = {tuple(i):j for i,j in zip(get_sent_pairs(sents_indexes), sr)}
    pred_segment_indices = get_segments(tau=tau, sent_indexes=sents_indexes, sr_dict=sr_dict)
    pred_paragraphs = [sents[block[0]:block[-1]+1] for block in pred_segment_indices]
    
    return sents, real_paragraphs, pred_segment_indices, pred_paragraphs


def segmentize_by_clustering(path_to_text, model):  
    
    freqs = joblib.load("freqs.pkl")
    
    with open(path_to_text, "r") as file:
        file = file.readlines()
        
    paragraphs = [line for line in file if not line.startswith("#") and line != "\n"]
    
    real_paragraphs = []
    sents = []
    for par in paragraphs:
        sent_list = [_.text for _ in sentenize(par)]
        real_paragraphs.append(sent_list)
        sents.extend(sent_list)
        
    tokens = []
    m = Mystem()
    print("Lemmatizing sentences")
    for sent in tqdm(sents):
        t = [_.text for _ in tokenize(sent)]
        lemmas = []
        for token in t:
            lemma = lemmatize(token, morph=m)
            if lemma != "":
                lemmas.append(lemma)
        tokens.append(lemmas)
    
    emb_dicts = get_embedding_dicts(tokens, model=model)
    dict_pairs = get_sent_pairs(emb_dicts)
    
    # count similarity scores for every pair of words in two sents
    sim_scores = []
    for k, pair in enumerate(dict_pairs):
    
        sent_1_number = str(k)
        sent_2_number = str(k+1)
    
        words_1 = [w for w in list(pair[0].keys()) if w not in stop_words and w not in punct]
        words_2 = [w for w in list(pair[1].keys()) if w not in stop_words and w not in punct]
        nodes_1 = [i + "_" + sent_1_number for i in words_1]
        nodes_2 = [i + "_" + sent_2_number for i in words_2]
        
        cur_sim_scores = {}
        for i,w_1 in enumerate(words_1):
            for j,w_2 in enumerate(words_2):
                weight = model.similarity(w_1, w_2)
                cur_sim_scores[(nodes_1[i], nodes_2[j])] = weight
        sim_scores.append(cur_sim_scores)
    
    # for every word take top-3 similar words
    top_3_sim_scores = []
    for d in sim_scores:
        words = list(set([k[0] for k in d.keys()]))
        for word in words:
            values = {k[1]:v for k,v in d.items() if k[0] == word}
            k = Counter(values) 
            high = {k:v for k,v in k.most_common(3)} 
            for k,v in high.items():
                if v > 0:
                    top_3_sim_scores.append((word, k, {"weight":v})) 
                    
    # build a graph with corresponding weights
    G = nx.Graph()
    G.add_edges_from(top_3_sim_scores)
  
    # cluster words
    partition = best_partition(G)
    clusters = {k:[] for k in list(set(partition.values()))}
    for k,v in partition.items():
        clusters[v].append(k)
        
    sent_clusters = {k:[i.split("_")[-1] for i in v] for k,v in clusters.items()}
    
    # get sentence distirubution over clusters
    sent_distribution_over_clusters = {}
    for k,v in sent_clusters.items():
        d = Counter(v)
        clust_len = sum(d.values())
        values = d.values()
        new_values = np.divide(np.array(list(values)),clust_len)
        new_d = dict(zip(d.keys(), list(new_values)))
        sent_distribution_over_clusters[k] = new_d
     
    
    # for every sentence pick up cluster with highest score
    final_clusters = {i:[] for i in list(sent_distribution_over_clusters.keys())}
    sents_indexes = [i for i in range(len(sents))]
    
    for indx in sents_indexes:
        max_sent_score = 0
        max_sent_cluster = 0
        for cluster in sent_distribution_over_clusters:
            d = sent_distribution_over_clusters[cluster]
            if str(indx) not in d:
                continue
            if d[str(indx)] > max_sent_score:
                max_sent_score = d[str(indx)]
                max_sent_cluster = cluster
        final_clusters[max_sent_cluster].append(indx)
    
    final_clusters = [i for i in sorted(list(final_clusters.values())) if i != []]

    pred_segment_indices = []
    for i in range(len(final_clusters)):
        if final_clusters[i][0] < final_clusters[i-1][-1] and i != 0:
            del pred_segment_indices[-1]
            pred_segment_indices.append(sorted(final_clusters[i]+final_clusters[i-1]))
        else:
            pred_segment_indices.append(final_clusters[i])
    
    pred_paragraphs = [sents[block[0]:block[-1]+1] for block in pred_segment_indices]
    
    return sents, real_paragraphs, pred_segment_indices, pred_paragraphs