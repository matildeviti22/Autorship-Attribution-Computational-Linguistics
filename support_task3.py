# LINGUISTICA COMPUTAZIONALE II -- MATILDE VITI
# support_task3.py

#import sqlite3
import numpy as np


def load_word_embeddings(src_path):
    """Carica gli embeddings da un file e li memorizza in un dizionario"""
    embeddings = dict()
    for line in open(src_path, 'r'):
        line = line.strip().split('\t')
        word = line[0]
        embedding = line[1:]
        embedding = [float(comp) for comp in embedding] # convertiamo le componenti dell'embedding in float
        embeddings[word] = np.asarray(embedding) # trasformiamo la lista delle componenti in un vettore di numpy
    return embeddings


def aggregate_mean(emb_list):
    """
    Calcola il vettore medio degli embeddings di un documento
    """
    total_vec = np.sum(emb_list, axis=0)
    mean_vec = total_vec / len(emb_list)
    return mean_vec


def aggregate_sum(emb_list):
    """
    Calcola la somma dei word embeddings di un documento
    """
    total_vec = np.sum(emb_list, axis=0)
    return total_vec


def aggregate_max(emb_list):
    """
    Calcola il massimo elemento per elemento tra gli embeddings
    """
    max_vec = np.max(emb_list, axis=0)
    return max_vec


def build_document_embedding(tokens, emb_dict, emb_dim, allowed_pos, method):
    """
    Estrae gli embeddings delle parole di un documento e li aggrega

    Args:
        tokens (list): lista di token con attributi `word` e `pos`
        emb_dict (dict): dizionario embeddings (word -> np.ndarray)
        emb_dim (int): dimensione dei vettori
        allowed_pos (list): POS da considerare
        method (str): tipo di aggregazione ('mean', 'sum', 'max')

    Returns:
        np.ndarray: embedding finale del documento
    """

    doc_vectors = []

    for tok in tokens:
        token_word = tok.word
        token_pos = tok.pos

        if token_word in emb_dict and token_pos in allowed_pos:
            doc_vectors.append(emb_dict[token_word])

    if len(doc_vectors) == 0:
        return np.zeros(emb_dim)

    if method == "mean":
        return aggregate_mean(doc_vectors)
    elif method == "sum":
        return aggregate_sum(doc_vectors)
    elif method == "max":
        return aggregate_max(doc_vectors)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")