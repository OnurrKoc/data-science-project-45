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

import re

import numpy as np


# Basit İngilizce stopword listesi (ön işleme için kullanabilirsin)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "this", "that", "these", "those", "it", "its", "his", "her",
    "their", "they", "he", "she", "we", "you", "i", "him", "them",
    "after", "ahead", "amid", "across", "into", "now", "can",
}


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
    pass


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
    pass


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
    pass


def train_classifier(pipe, X, y):
    """
    Pipeline'ı fit et ve döndür.

    İpucu: pipe.fit(X, y); return pipe
    """
    pass


def classify(pipe, text):
    """
    Tek bir metin için kategori tahmini döndür (str).

    İpucu: pipe.predict([text])[0]
    """
    pass


def evaluate(pipe, X, y):
    """
    Verilen X, y üzerinde accuracy döndür (float).

    İpucu:
    - from sklearn.metrics import accuracy_score
    - y_pred = pipe.predict(X); return accuracy_score(y, y_pred)
    """
    pass


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
    pass


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
    pass


# 5. Modeli kaydet / yükle — joblib
def save_model(pipe, path):
    """
    Eğitilmiş pipeline'ı diske kaydet (joblib veya pickle). path'i döndür.

    İpucu:
    - import joblib
    - joblib.dump(pipe, path); return path
    """
    pass


def load_model(path):
    """
    Diskten pipeline yükle ve döndür.

    İpucu:
    - import joblib
    - return joblib.load(path)
    """
    pass


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
    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 NLP Capstone Pipeline Sonuçları:")
    print(f"  Accuracy          : {result['accuracy']:.4f}")
    print(f"  Örnek tahmin      : {result['sample_prediction']}")
    print(f"  Çıkarılan varlıklar: {result['entities']}")
    print(f"  Arama sonuçları   :")
    for doc, score in result["search_results"]:
        print(f"    [{score:.3f}] {doc}")
