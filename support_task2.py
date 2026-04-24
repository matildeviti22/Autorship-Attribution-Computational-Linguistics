# LINGUISTICA COMPUTAZIONALE II -- MATILDE VITI
# support_task2.py

import json

def load_json(json_path):
    """Carica un file JSON e restituisce il contenuto come oggetto Python."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def ngram_from_tokens(doc, tipo, n):
    """Estrae n-gram di parole, lemmi o POS da un documento."""
    if tipo == 'word':
        elems = doc.get_words()
    elif tipo == 'lemma':
        elems = doc.get_lemmas()
    elif tipo == 'pos':
        elems = doc.get_pos()
    else:
        raise ValueError("Tipo non valido: deve essere 'word', 'lemma' o 'pos'")

    ngram_count = {}
    for i in range(len(elems) - n + 1):
        key = f"{tipo.upper()}_{n}_" + "_".join(elems[i:i+n])
        ngram_count[key] = ngram_count.get(key, 0) + 1
    return ngram_count


def ngram_from_chars(doc, n):
    """Estrae n-gram di caratteri da un documento."""
    text = " ".join(doc.get_words())
    ngram_count = {}
    for i in range(len(text) - n + 1):
        key = f"CHAR_{n}_" + text[i:i+n]
        ngram_count[key] = ngram_count.get(key, 0) + 1
    return ngram_count


def normalize_ngrams_dict(ngram_dict, lunghezza):
    """Normalizza i valori dividendo per la lunghezza totale."""
    return {k: v / lunghezza for k, v in ngram_dict.items()}


def extract_all_ngrams(docs, config):
    """
    Estrae e normalizza n-gram secondo la lista di configurazioni.
    config = [(tipo, n), ...] dove tipo in 'word', 'lemma', 'pos', 'char'
    """
    for doc in docs:
        all_features = {}
        for tipo, n in config:
            if tipo == 'char':
                ng = ngram_from_chars(doc, n)
                ng = normalize_ngrams_dict(ng, doc.get_num_chars())
            else:
                ng = ngram_from_tokens(doc, tipo, n)
                if tipo == 'word':
                    lung = doc.get_num_tokens()
                elif tipo == 'lemma':
                    lung = doc.get_num_lemmas()
                elif tipo == 'pos':
                    lung = doc.get_num_pos()
                ng = normalize_ngrams_dict(ng, lung)
            all_features.update(ng)
        doc.features = all_features


def split_dataset(docs):
    """Divide i documenti in train, val e test secondo doc.split."""
    train_feats, train_labels = [], []
    val_feats, val_labels = [], []
    test_feats, test_labels = [], []

    for doc in docs:
        if doc.split == 'train':
            train_feats.append(doc.features)
            train_labels.append(doc.author)
        elif doc.split == 'test':
            test_feats.append(doc.features)
            test_labels.append(doc.author)
        else:
            val_feats.append(doc.features)
            val_labels.append(doc.author)
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels