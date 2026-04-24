# LINGUISTICA COMPUTAZIONALE II -- MATILDE VITI
# support_preprocessing.py


from pathlib import Path
import re
import logging
from collections import Counter
import pandas as pd
import random

### Caricamento dei file di testo e gestione della struttura delle cartelle
def load_raw_texts(root_folder: str):
    root = Path(root_folder)
    
    for author_folder in root.iterdir():
        if not author_folder.is_dir():
            continue
        
        author = author_folder.name
        
        for file_path in author_folder.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"\nSto processando '{file_path.name}' di {author}")
            yield author, file_path.stem, text



def _is_real_start(lines: list[str], from_index: int, heading_re: re.Pattern, threshold: int = 4) -> bool:
    """
Questa funzione verifica se una riga che sembra un titolo (es. Chapter I, Volume I) corrisponde davvero all'inizio della narrazione oppure se si trova ancora dentro un indice.

1. Prende le 40 righe successive a quella candidata.
2. Elimina le righe vuote.
3. Controlla quante delle prime 20 righe non vuote sono heading usando la var heading_re che controlla se trova righe che iniziano con Volume, Book, Part o Chapter
4. Se il numero di heading è inferiore a una soglia (`threshold`), considera la riga un possibile inizio reale del libro.

Idea: 
Negli indici Gutenberg compaiono molte righe tipo Chapter I, Chapter II, Chapter III... una dopo l’altra.  
All’inizio reale del libro invece trovi solo una o poche intestazioni, seguite subito dal testo narrativo.
"""
    next_nonempty = [
        x.strip()
        for x in lines[from_index: from_index + 40] # guarda le prossime 40 righe dopo il possibile inizio
        if x.strip() 
    ] # lista delle prossime righe non vuote
    heading_count = sum(
        1 for t in next_nonempty[:20] # guarda solo le prime 20 righe non vuote e somma quante sembrano essere heading
        if heading_re.match(t)
    )

    return heading_count < threshold # se il numero di heading è inferiore alla soglia, allora probabilmente non siamo dentro un indice



def find_narrative_start(lines: list[str]) -> list[str]: # parte già dal testo diviso in righe e restituisce solo le righe a partire dall'inizio narrativo vero
    """

Questa funzione individua il punto in cui inizia davvero la narrazione del libro, rimuovendo il materiale iniziale (front matter) come:

- intestazioni Gutenberg
- dediche
- indici
- informazioni editoriali

Strategia:

1. Parte da testo già diviso in righe.
2. Cerca nelle prime 800 righe una sezione chiamata "Contents" o "Table of contents".
3. Se la trova, scansiona le righe successive cercando un titolo plausibile con l'uso obbligatorio di I, one o 1
4. Per ogni candidato usa is_real_start() per verificare che non sia ancora dentro l’indice.
5. Quando trova un candidato valido, restituisce il testo da quel punto fino alla fine.

Fallback:
Se non esiste una sezione Contents, la funzione prova comunque a trovare un possibile inizio scansionando tutto il testo.

Se non trova nulla, restituisce il testo originale e stampa un warning.
    """
    contents_re = re.compile(
        r"^\s*(contents|table of contents)\s*[:.]?\s*$", # corrisponde a "Contents" o "Table of Contents" con eventuali spazi e punteggiatura: se trova Contents, potrebbe essere un indice!
        re.IGNORECASE
    )

    heading_re = re.compile(
        r"^\s*(volume|book|part|chapter)\b", # questa riconosce un vero e proprio heding da Indice, se trova righe che iniziano con Volume, Book, Part o Chapter, è molto probabile che siamo dentro un indice 
        re.IGNORECASE
    )

    real_start_re = re.compile(
        r"^\s*(volume|book|part|chapter|preface|introduction|prologue)\b(?:\s+(i|1|one))", # con questa, invece, cerchiamo un vero inizio narrativo, con l'uso obbligatorio di 1, I o "One" dopo Volume, Book, Part, Chapter, Preface, Introduction o Prologue: se troviamo questo pattern, è molto probabile che siamo davvero all'inizio della narrazione
        re.IGNORECASE
    )

    # 1) Cerca "Contents" nelle prime 800 righe (Numero indicativo, perché di solito si trova all'inizio del libro)
    contents_pos = next(
        (i for i, ln in enumerate(lines[:800]) if contents_re.match(ln.strip())),
        None
    )

    # 2) Se trova Contents, prova da lì
    if contents_pos is not None:
        for i in range(contents_pos + 1, len(lines)):
            s = lines[i].strip()
            if s and real_start_re.match(s) and _is_real_start(lines, i, heading_re): # Se non è una riga vuota, se sembra un vero inizio narrativo (es. "Chapter 1") e se non siamo dentro un indice e chiama _is_real_start per verificare che non siamo dentro un indice
                return lines[i:] # se trova un vero inizio narrativo dopo Contents, restituisce il testo a partire da lì

    # 3) Fallback: cerca sull'intero testo
    for i, line in enumerate(lines):
        s = line.strip()
        if s and real_start_re.match(s) and _is_real_start(lines, i, heading_re):
            return lines[i:]

    logging.warning("find_narrative_start: nessun inizio trovato, restituisco testo intero")
    return lines

#     print(f"Lunghezza testo: {len(text)} \n")



def _normalize_line(line: str) -> str:
    # prende una singola riga e la normalizza per poter fare confronti affidabili
    s = line.strip().upper()
    s = re.sub(r"^[\[\(\{'\"]+", "", s) # rimuove i caratteri di apertura
    s = re.sub(r"[\]\)\}'\"\.:\-;!,\s]+$", "", s) # rimuove i caratteri di chiusura
    return s



def trim_footer(lines: list[str]) -> list[str]:
    """
    Questa funzione rimuove il footer editoriale dai testi dei libri.

    Il procedimento è il seguente:

    1. Cerca possibili marker di fine narrativa come "THE END" o "FINIS"
    (gestendo anche il caso in cui "THE" e "END" siano su due righe separate).

    2. Se trova uno di questi marker, controlla le righe successive alla ricerca di
    segnali editoriali (ad esempio *printer, publisher, catalog, london, press*).
    Se presenti, considera quel punto come la fine del libro e taglia il testo.

    3. Se non trova marker narrativi, utilizza come fallback il marker standard  
    **"END OF THE PROJECT GUTENBERG EBOOK"**.

    4. Se nessun marker viene trovato, il testo viene restituito senza modifiche.

    Per rendere il riconoscimento più robusto, le righe vengono prima normalizzate con _normalize_line
    (rimozione di punteggiatura semplice e conversione in maiuscolo)."""

    editorial_signals = [
        "ESTABLISHED", "PRINTER", "PRINTERS", "PUBLISHER", "PUBLISHERS",
        "CLASSICS", "UNIFORM WITH", "SAME PRICE", "CATALOG", "CATALOGUE",
        "ILLUSTRATION", "LONDON", "PRINTED BY", "PRINTED IN", "COPYRIGHT",
        "COPYRIGHTED", "PRESS"
    ] # se dopo un possibile marker di fine troviamo parole come queste, è molto probabile che siamo davvero alla fine del testo narrativo e che il resto sia materiale editoriale o pubblicitario

    # A) Cerca marker narrativi finali: THE END, FINIS, THE / END
    end_candidates = []

    for i in range(len(lines)):
        s = _normalize_line(lines[i]) # normalizza la riga per confronti affidabili (es. "The End" diventa "THE END", e rimuove eventuali caratteri di punteggiatura o spazi all'inizio o alla fine)

        if s == "THE END":
            end_candidates.append(i) 

        elif s == "FINIS":
            end_candidates.append(i) # In testi più vecchi, il finale è indicato con "Finis" invece di "The End", quindi l'ho considerato un possibile marker di fine

        elif s == "THE" and i + 1 < len(lines): # In alcuni casi "The End" è diviso su due righe
            s_next = _normalize_line(lines[i + 1])
            if s_next == "END":
                end_candidates.append(i + 1)

    # Usa l'ultimo candidato plausibile (se ci sono più The End, viene preso l'ultimo)
    if end_candidates:
        cand = end_candidates[-1]
        after = "\n".join(lines[cand + 1:cand + 1 + 200]).upper() # guarda le 200 righe dopo il possibile marker di fine e normalizza

        if any(sig in after for sig in editorial_signals) or not after.strip(): # se dopo il possibile marker di fine troviamo parole che indicano materiale editoriale o pubblicitario, oppure se non ci sono righe significative dopo, allora è molto probabile che siamo davvero alla fine del testo narrativo
            return lines[:cand + 1] # dunque, se dopo "THE END" abbiamo parole come "Printed by" o "publisher", ecc. oppure se non ci sono righe significative dopo, allora è molto probabile che siamo davvero alla fine del testo narrativo e che il resto sia materiale editoriale o pubblicitario, quindi restituisce il testo fino a quel punto (incluso "THE END"!!!)

    # B) Marker standard di Project Gutenberg
    gutenberg_end_re = re.compile(
        r"(\*{3}\s*)?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", # Prende sia "*** END OF THIS PROJECT GUTENBERG EBOOK" che "END OF THE PROJECT GUTENBERG EBOOK", ecc. (con eventuali spazi e asterischi all'inizio)
        re.IGNORECASE
    )
    end_idx = None
    for i, ln in enumerate(lines):
        if gutenberg_end_re.search(ln):
            end_idx = i
            break

    if end_idx is not None:
        return lines[:end_idx]

    logging.warning("trim_footer: nessun marker di fine trovato, restituisco testo intero")
    return lines




def remove_illustrations(text: str) -> str:
    """Questa funzione rimuove dal testo i blocchi editoriali che rappresentano illustrazioni, tipicamente nel formato: [Illustration...]  
- `re.IGNORECASE` permette di riconoscere sia `Illustration` che `illustration`.
- `re.DOTALL` consente di catturare anche blocchi su più righe.
"""
    return re.sub(r"\[\s*Illustration.*?\s*\]", "", text, flags=re.IGNORECASE | re.DOTALL)




def remove_footnotes(text: str) -> str:
    # Questa funzione rimuove le note a piè di pagina e i riferimenti bibliografici 

    # Note lunghe
    text = re.sub( 
        r"\[(?:Note|Footnote|Footnotes|Ref)\s*:?.*?\]", # corrisponde a note che iniziano con "Note", "Footnote", "Footnotes" o "Ref", con eventuale ":" dopo e qualsiasi testo fino alla chiusura della parentesi
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL # re.IGNORECASE per non fare distinzione tra maiuscole e minuscole, re.DOTALL per far sì che il punto corrisponda anche a newline, in modo da poter rimuovere note che si estendono su più righe
    )

    text = re.sub(r"\[\d+\]", "", text) # rimuove riferimenti brevi come [1], [2], ecc.
    text = re.sub(r"\{\d+\}", "", text) # rimuove gli stessi riferimenti ma con {}

    # Asterischi isolati
    text = re.sub(r"\s\*\s", " ", text) # rimuove asterischi isolati circondati da spazi, che spesso indicano note a piè di pagina

    return text



def remove_gutenberg_header(text: str) -> str: 
    # Questa funzione rimuove l'intestazione standard di Project Gutenberg, che inizia con una riga del tipo "*** START OF THIS PROJECT GUTENBERG EBOOK ... ***".
    start_re = re.compile(
        r"\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*{3}",
        re.IGNORECASE | re.DOTALL
    )
    m_start = start_re.search(text)
    if m_start:
        return text[m_start.end():]
    return text




def normalize_dashes(text: str) -> str: # Dopo aver visto che ci sono molti usi impropri di trattini nei testi, ho deciso di creare questa funzione per normalizzarli e gestirli in modo più efficace. 
    # L'obiettivo è unificare le varie forme di dash e trasformare sequenze di trattini in un formato più standard, come l'em dash (—) 

    # !!! Non ho toccato i trattini normali usati per parole composte o per indicare pause brevi !!!

    # Unifica alcune varianti Unicode al dash lungo standard
    text = text.replace("–", "—")
    text = text.replace("―", "—")

    # Trasforma sequenze ASCII di almeno due trattini in un solo em dash
    text = re.sub(r"-{2,}", "—", text)

    # Evita ripetizioni accidentali di em dash
    text = re.sub(r"—{2,}", "—", text)

    return text




def remove_isolated_brackets(text: str) -> str:
    # Questa funzione rimuove le righe che contengono solo parentesi quadre o tonde isolate, che spesso appaiono nei testi di Project Gutenberg come artefatti editoriali o segnaposto per elementi mancanti.
    import re
    return re.sub(r"^\s*[\[\]]\s*$", "", text, flags=re.MULTILINE)



def remove_italic_markers(text: str) -> str: # Funzione per rimuovere i marker di italic rappresentati da underscore
    return re.sub(r"_([^_]+)_", r"\1", text)



def clean_gutenberg(text: str) -> str:
    # Questa funzione esegue una pulizia approfondita del testo di Project Gutenberg, rimuovendo elementi non narrativi e normalizzando il formato.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) rimuove header Gutenberg
    text = remove_gutenberg_header(text)

    # 2) divide in righe
    lines = text.splitlines()

    # 3) trova inizio narrativo
    lines = find_narrative_start(lines)

    # 4) taglia footer
    lines = trim_footer(lines)

    # 5) ricompone il testo
    text = "\n".join(lines)

    # 6) altre pulizie
    text = remove_illustrations(text)
    text = remove_footnotes(text)
    text = normalize_dashes(text)
    text = remove_isolated_brackets(text)
    text = remove_italic_markers(text)

    # 7) normalizzazione spazi
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text



def inspect_noise(text: str):

    # Questa funzione verifica la qualità della pulizia, nel file di preprocessing la uso dopo l'applicazione di clean_gutenberg per vedere se sono ancora presenti rimasugli di URL, email, marker editoriali, note a piè di pagina, ecc.
    lines = text.splitlines()

    patterns = {
        "url": r"(https?://\S+|www\.\S+)",
        "email": r"\b\S+@\S+\.\S+\b",
        "gutenberg": r"project gutenberg",
        "metadata": r"^(title:|author:|release date:|language:|credits:|other information and formats:)",
        "illustration": r"^\s*\[illustration.*",
        "pg_marker": r"\*\*\*\s*(start|end)\s+of\s+(the|this)\s+project gutenberg.*",
        "final_markers": r"^\s*(the\s+end|finis|the\s*[\n\r]+end)\s*$",
        "footnote": r"(\[\d+\]|\{\d+\}|\[(note|footnote|ref)\s*:?)",
        "decorative": r"^\s*([*_=-]\s*){3,}$",
        "isolated_bracket": r"^\s*[\[\]]\s*$",
        "paratext": r"^\s*(cover|frontispiece|plate|caption)\s*$",
        "italic": r"_([^_]+)_",
    }

    counts = {key: 0 for key in patterns}

    for line in lines:
        for key, pat in patterns.items():
            if re.search(pat, line, flags=re.IGNORECASE):
                counts[key] += 1

    # pattern con valore > 0
    non_zero = {k: v for k, v in counts.items() if v > 0}

    # caso speciale: solo final_markers = 1
    if not non_zero or (len(non_zero) == 1 and counts["final_markers"] == 1):
        print("✓ CLEAN TEXT – no noise detected (final marker preserved)")
    else:
        for k, v in non_zero.items():
            if k != "final_markers":
                print(f"{k:20} : {v}")

    return counts


def split_into_paragraphs(text: str):
    """
    Divide il testo di un libro in paragrafi e seleziona solo quelli che rispettano
    il vincolo di lunghezza richiesto (50-100).

    Procedura:

    1. Il testo viene inizialmente suddiviso in blocchi utilizzando una o più righe
       vuote come separatori di paragrafo.

    2. Per ogni blocco individuato:
       - i ritorni a capo interni al paragrafo vengono sostituiti con uno spazio
       - gli spazi multipli vengono normalizzati in un singolo spazio.

    3. I paragrafi vuoti o composti solo da spazi vengono scartati.

    4. Il paragrafo viene tokenizzato in modo semplice tramite split sugli spazi.

    5. Vengono mantenuti solo i paragrafi che contengono tra 50 e 100 token,
       come richiesto dalle specifiche del progetto.
    """
    # divide il testo in blocchi usando una o più righe vuote come separatori
    raw_paragraphs = re.split(r"\n\s*\n+", text.strip()) # trova separatori tipo \n\n
    paragraphs = []

    for p in raw_paragraphs:
        # sostituisce i ritorni a capo interni al paragrafo con uno spazio
        p = re.sub(r"\s*\n\s*", " ", p)

        # normalizza gli spazi multipli
        p = re.sub(r"\s{2,}", " ", p).strip() # sistema gli spazzi doppi o tripli o + con un solo spazio

        if not p: # salta paragrafi vuoti o che diventano vuoti dopo la normalizzazione
            continue

        tokens = p.split() # divide il paragrafo in token usando lo spazio come delimitatore

        if 50 <= len(tokens) <= 100:  # filtra i paragrafi mantenendo solo quelli che hanno un numero di token compreso tra 50 e 100
            paragraphs.append(p)

    return paragraphs


def undersample_splits(dataset: dict, max_per_author: dict, seed: int = 42) -> dict:
    """
    Questa funzione esegue un undersampling bilanciato per autore sui dataset di training, validation e test, limitando il numero di campioni per autore a un massimo specificato."""
    for split_name, max_n in max_per_author.items(): # per ogni split (train, val, test) e il corrispondente numero massimo di campioni per autore
        if max_n is None:
            continue
        by_author = {} # crea un dizionario per raggruppare i campioni per autore
        for sample in dataset[split_name]: # per ogni campione nello split corrente
            by_author.setdefault(sample["author"], []).append(sample) # raggruppa i campioni per autore, creando una lista di campioni per ogni autore
        balanced = []
        for author, samples in by_author.items(): 
            random.seed(seed) # imposta il seed per la riproducibilità
            balanced.extend(random.sample(samples, min(max_n, len(samples)))) # per ogni autore, seleziona casualmente un numero di campioni fino al massimo specificato (o tutti se ce ne sono meno di max_n) e li aggiunge alla lista bilanciata
        dataset[split_name] = balanced # sostituisce lo split originale con quello bilanciato
    return dataset