"""
DS-45 — Uçtan Uca NLP Pipeline (Capstone)

İleri NLP dersinde öğrendiğin TÜM araçları TEK bir pipeline'da birleştiriyorsun:
metin ön işleme → TF-IDF + LogisticRegression sınıflandırma → spaCy ile varlık
çıkarımı (NER) → sentence-transformers ile semantik arama → modeli diske kaydet/yükle.

Senaryo: bir haber/medya platformunda NLP mühendisisin. Gelen kısa metinleri
kategorilere (sport / business / tech) ayırıyor, içlerindeki kişi/kurum/yer
isimlerini çıkarıyor ve "anlamca en yakın" dokümanları semantik arama ile
buluyorsun. Hepsi tek bir uçtan uca akışta.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`

NOT: spaCy ve sentence-transformers modelleri büyüktür; ilk yüklemede birkaç
saniye sürer. Bu yüzden çözümde modelleri modül-seviyesinde bir kez yükleyip
cache'lemek mantıklıdır (zorunlu değil).
"""

import os
import re
import pickle

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_DATASET = [
    # sport
    ("The football team won the championship final after a dramatic penalty shootout.", "sport"),
    ("Serena Williams advanced to the tennis semifinal with a powerful serve.", "sport"),
    ("The basketball player scored forty points in last night's game.", "sport"),
    ("The marathon runner broke the world record in the Olympic stadium.", "sport"),
    ("The coach praised his players after a stunning comeback victory.", "sport"),
    ("Cristiano Ronaldo scored a hat-trick in the league match on Saturday.", "sport"),
    ("The cricket team celebrated their tournament win with the trophy.", "sport"),
    ("The swimmer earned a gold medal in the freestyle relay event.", "sport"),
    ("Fans cheered as the goalkeeper saved the decisive penalty kick.", "sport"),
    ("The boxing champion defended his title in a twelve round bout.", "sport"),
    ("The rugby squad trained hard ahead of the World Cup quarterfinal.", "sport"),
    ("The cyclist sprinted to the finish line to win the mountain stage.", "sport"),

    # business
    ("The company reported record quarterly profits and raised its dividend.", "business"),
    ("Investors reacted to the central bank's decision on interest rates.", "business"),
    ("The startup raised fifty million dollars in a new funding round.", "business"),
    ("Shares of the technology firm surged after strong earnings results.", "business"),
    ("The merger between the two banks created a financial giant.", "business"),
    ("The stock market fell sharply amid fears of rising inflation.", "business"),
    ("The CEO announced a restructuring plan to cut operating costs.", "business"),
    ("Oil prices climbed as demand recovered across global markets.", "business"),
    ("The retailer expanded into new markets to boost annual revenue.", "business"),
    ("Analysts forecast steady growth for the consumer goods sector.", "business"),
    ("The corporation acquired a rival to strengthen its market share.", "business"),
    ("Quarterly sales exceeded expectations, lifting the firm's valuation.", "business"),

    # tech
    ("The new smartphone features a faster chip and an improved camera.", "tech"),
    ("Engineers released a software update to patch a security vulnerability.", "tech"),
    ("The artificial intelligence model can generate realistic images from text.", "tech"),
    ("The cloud platform now supports faster data processing at scale.", "tech"),
    ("Developers built an app using the latest machine learning framework.", "tech"),
    ("The laptop comes with a high resolution display and longer battery life.", "tech"),
    ("The company unveiled a self driving car powered by deep learning.", "tech"),
    ("A new programming language promises safer and faster code execution.", "tech"),
    ("The robotics lab demonstrated a humanoid robot that can walk and grasp.", "tech"),
    ("The chipmaker announced a powerful processor for data centers.", "tech"),
    ("The operating system update brings new privacy and battery features.", "tech"),
    ("Researchers trained a neural network to translate dozens of languages.", "tech"),
]

# Basit İngilizce stopword listesi (ön işleme için kullanabilirsin)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "this", "that", "these", "those", "it", "its", "his", "her",
    "their", "they", "he", "she", "we", "you", "i", "him", "them",
    "after", "ahead", "amid", "across", "into", "now", "can",
}

_SPACY_NLP = None
_ST_MODEL = None


def _get_spacy():
    """spaCy en_core_web_sm modelini bir kez yükle, sonra cache'le."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def _get_st_model():
    """sentence-transformers all-MiniLM-L6-v2 modelini bir kez yükle, cache'le."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

# 0. Gömülü veri seti
def load_data():
    """
    Koda gömülü küçük bir veri seti döndür: ~30-40 İngilizce kısa metin ve
    3-4 kategori etiketi (örn. 'sport', 'business', 'tech'). Her kategoride
    en az 2 örnek olsun.

    Returns:
        tuple: (X, y)
            X: list[str] — metinler
            y: list[str] — kategori etiketleri

    İpucu:
    - Kendi cümlelerini yaz (her kategoriden ~10-12 cümle).
    - sport: maç, gol, şampiyon... / business: kâr, hisse, banka... /
      tech: yazılım, çip, yapay zekâ...
    - X ve y aynı uzunlukta olmalı.
    """
    X = [text for text, _ in _DATASET]
    y = [label for _, label in _DATASET]
    return X, y


# 1. Ön işleme
def preprocess(text):
    """
    Metni temizle:
    1. Küçük harfe çevir.
    2. Regex ile harf olmayan karakterleri (rakam, noktalama) boşluğa çevir.
    3. İngilizce stopword'leri at (_STOPWORDS kullanabilirsin).
    4. Kalan kelimeleri tek boşlukla birleştirip döndür.

    Args:
        text: str

    Returns:
        str: temizlenmiş metin

    İpucu:
    - text = text.lower()
    - text = re.sub(r"[^a-z\\s]", " ", text)
    - tokens = [t for t in text.split() if t not in _STOPWORDS]
    - return " ".join(tokens)
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)


# 2. Sınıflandırıcı pipeline — TF-IDF + LogisticRegression
def build_classifier():
    """
    sklearn Pipeline kur:
    - 'tfidf': TfidfVectorizer  (istersen preprocessor=preprocess geçebilirsin)
    - 'clf':   LogisticRegression(max_iter=1000, random_state=42)

    Returns:
        sklearn.pipeline.Pipeline

    İpucu:
    - from sklearn.pipeline import Pipeline
    - from sklearn.feature_extraction.text import TfidfVectorizer
    - from sklearn.linear_model import LogisticRegression
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def train_classifier(pipe, X, y):
    """
    Pipeline'ı fit et ve döndür.

    İpucu: pipe.fit(X, y); return pipe
    """
    pipe.fit(X, y)
    return pipe


def classify(pipe, text):
    """
    Tek bir metin için kategori tahmini döndür (str).

    İpucu: pipe.predict([text])[0]
    """
    return pipe.predict([text])[0]


def evaluate(pipe, X, y):
    """
    Verilen X, y üzerinde accuracy döndür (float).

    İpucu:
    - from sklearn.metrics import accuracy_score
    - y_pred = pipe.predict(X); return accuracy_score(y, y_pred)
    """
    y_pred = pipe.predict(X)
    return accuracy_score(y, y_pred)


# 3. Named Entity Recognition — spaCy
def extract_entities(text):
    """
    spaCy en_core_web_sm modeli ile metindeki varlıkları (entity) çıkar.

    Args:
        text: str

    Returns:
        list[tuple]: [(entity_text, entity_label), ...]
        Örn. [('Apple', 'ORG'), ('California', 'GPE')]

    İpucu:
    - import spacy
    - nlp = spacy.load("en_core_web_sm")   # bir kez yükle, cache'le (global)
    - doc = nlp(text)
    - return [(ent.text, ent.label_) for ent in doc.ents]
    """
    nlp = _get_spacy()
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# 4. Semantik arama — sentence-transformers + cosine
def semantic_search(query, docs, top_k=3):
    """
    sentence-transformers 'all-MiniLM-L6-v2' modeli ile semantik arama yap.
    query ve docs'u embed et, cosine similarity'ye göre en yakın top_k
    dokümanı skoruyla döndür (skor AZALAN sırada).

    Args:
        query: str
        docs: list[str]
        top_k: int (default 3)

    Returns:
        list[tuple]: [(doc, score), ...] — en fazla top_k adet, skor azalan

    İpucu:
    - from sentence_transformers import SentenceTransformer
    - model = SentenceTransformer("all-MiniLM-L6-v2")   # bir kez yükle, cache'le
    - doc_emb = model.encode(docs, normalize_embeddings=True)
    - q_emb = model.encode([query], normalize_embeddings=True)[0]
    - normalize edilince cosine = dot product → scores = doc_emb @ q_emb
    - order = np.argsort(scores)[::-1][:top_k]
    - return [(docs[i], float(scores[i])) for i in order]
    """
    model = _get_st_model()
    doc_emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = doc_emb @ q_emb
    order = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in order]


# 5. Modeli kaydet / yükle — joblib
def save_model(pipe, path):
    """
    Eğitilmiş pipeline'ı diske kaydet (joblib veya pickle). path'i döndür.

    İpucu:
    - import joblib
    - joblib.dump(pipe, path); return path
    """
    import joblib
    joblib.dump(pipe, path)
    return path


def load_model(path):
    """
    Diskten pipeline yükle ve döndür.

    İpucu:
    - import joblib
    - return joblib.load(path)
    """
    import joblib
    return joblib.load(path)


# 6. Uçtan uca pipeline
def run_pipeline():
    """
    Tüm aşamaları uçtan uca çalıştır:
    1. load_data
    2. build_classifier → train_classifier → evaluate (accuracy)
    3. classify ile bir örnek metni sınıflandır
    4. extract_entities ile bir metinden varlık çıkar
    5. semantic_search ile bir sorgu için ilgili dokümanları bul

    Returns:
        dict: {
            'accuracy': float,
            'sample_prediction': str,
            'entities': list[tuple],
            'search_results': list[tuple]
        }
    """
    X, y = load_data()

    # Sınıflandırma
    pipe = build_classifier()
    pipe = train_classifier(pipe, X, y)
    acc = evaluate(pipe, X, y)
    sample_pred = classify(pipe, "The team won the game with a last minute goal.")

    # NER
    entities = extract_entities(
        "Apple unveiled a new iPhone in California while Serena Williams won in London."
    )

    # Semantik arama
    results = semantic_search("a player scored a goal in the match", X, top_k=3)

    return {
        "accuracy": acc,
        "sample_prediction": sample_pred,
        "entities": entities,
        "search_results": results,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 NLP Capstone Pipeline Sonuçları:")
    print(f"  Accuracy          : {result['accuracy']:.4f}")
    print(f"  Örnek tahmin      : {result['sample_prediction']}")
    print(f"  Çıkarılan varlıklar: {result['entities']}")
    print(f"  Arama sonuçları   :")
    for doc, score in result["search_results"]:
        print(f"    [{score:.3f}] {doc}")
