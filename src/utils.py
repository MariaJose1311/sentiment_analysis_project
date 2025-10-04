import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

# ======================
# Semilla
# ======================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================
# Métricas
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    average = "binary" if max(labels) <= 1 else "macro"
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=average, zero_division=0)
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1)}

# ======================
# Limpieza de texto
# ======================
def limpiar_texto(texto, extra_stopwords=None):
    # Try to load NLTK Spanish stopwords; if missing, attempt download; if still missing, fallback
    try:
        stop_words = set(stopwords.words("spanish"))
    except Exception:
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words("spanish"))
        except Exception:
            # Fallback: use wordcloud STOPWORDS plus a minimal spanish stoplist
            stop_words = set(STOPWORDS)
            stop_words.update({
                "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las",
                "por", "un", "para", "con", "no", "una", "su", "al", "es", "lo",
                "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque"
            })

    stop_words.update(STOPWORDS)
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    palabras = [w for w in texto.split() if w not in stop_words and len(w) > 2]
    return " ".join(palabras)

# ======================
# Mapear rating a etiquetas
# ======================
def map_label(rating):
    return "no positiva" if rating <= 3 else "positiva"


# ======================
# Extracción de bigramas y frases resumen
# ======================
def extraer_bigrams(texts, top_n=10):
    """
    Extrae los bigramas más frecuentes de un iterable de textos limpios (strings).
    Devuelve una lista de tuplas (bigram, conteo) ordenada por conteo desc.
    """
    # Convert pandas Series or other iterables to a list of strings and filter empty values
    try:
        texts_list = [str(t) for t in list(texts) if pd.notna(t) and str(t).strip()]
    except Exception:
        texts_list = [str(t) for t in texts if t and str(t).strip()]

    if len(texts_list) == 0:
        return []
    try:
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='spanish')
        X = vec.fit_transform(texts_list)
    except Exception:
        # fallback without stop_words if language not recognized
        vec = CountVectorizer(ngram_range=(2, 2))
        X = vec.fit_transform(texts_list)

    sums = np.asarray(X.sum(axis=0)).ravel()
    if sums.size == 0:
        return []
    indices = np.argsort(sums)[::-1][:top_n]
    features = np.array(vec.get_feature_names_out())[indices]
    counts = sums[indices]
    return list(zip(features.tolist(), counts.tolist()))


def frases_para_resumen(bigrams_counts, top_k=3):
    """
    Convierte una lista de bigramas con conteos en frases cortas para incluir en el resumen.
    Entrada esperada: [('muy bueno', 12), ('buen servicio', 8), ...]
    Devuelve lista de strings en español.
    """
    if not bigrams_counts:
        return []
    frases = []
    for bigram, count in bigrams_counts[:top_k]:
        frases.append(f"Se menciona '{bigram}' {int(count)} veces")
    return frases
