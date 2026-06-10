import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_data, preprocess,
    build_classifier, train_classifier, classify, evaluate,
    extract_entities, semantic_search,
    save_model, load_model, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — ağır modeller / eğitim bir kez
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def trained_pipeline(data):
    X, y = data
    pipe = build_classifier()
    return train_classifier(pipe, X, y)


# 1. load_data — yapı
def test_load_data_structure(data):
    X, y = data
    assert isinstance(X, list) and isinstance(y, list)
    assert len(X) == len(y)
    # ~30-40 metin
    assert 30 <= len(X) <= 40
    assert all(isinstance(t, str) and t for t in X)


# 2. load_data — kategoriler (3-4 sınıf)
def test_load_data_categories(data):
    _, y = data
    cats = set(y)
    assert 3 <= len(cats) <= 4
    # Her kategoride en az iki örnek olmalı
    for c in cats:
        assert y.count(c) >= 2


# 3. preprocess — temizleme
def test_preprocess_lowercase_regex():
    out = preprocess("The COMPANY's profits ROSE 50%, amazing!!!")
    # Küçük harf, noktalama/rakam temizlenmiş
    assert out == out.lower()
    assert not any(ch.isdigit() for ch in out)
    assert "%" not in out and "!" not in out
    # Stopword 'the' atılmış olmalı
    assert "the" not in out.split()


# 4. preprocess — boş/anlamlı çıktı
def test_preprocess_keeps_content():
    out = preprocess("Engineers released a software update.")
    assert "engineers" in out
    assert "software" in out
    assert "update" in out


# 5. build_classifier — tip
def test_build_classifier_type():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipe = build_classifier()
    assert isinstance(pipe, Pipeline)
    names = dict(pipe.steps)
    assert 'tfidf' in names and 'clf' in names
    assert isinstance(names['tfidf'], TfidfVectorizer)
    assert isinstance(names['clf'], LogisticRegression)


# 6. train_classifier — fit edilebilir
def test_train_classifier_fits(trained_pipeline, data):
    X, _ = data
    preds = trained_pipeline.predict(X[:3])
    assert len(preds) == 3


# 7. classify — tek metin
def test_classify_single(trained_pipeline):
    pred = classify(trained_pipeline, "The athlete won the race at the stadium.")
    assert isinstance(pred, str)
    # Bilinen kategorilerden biri
    assert pred in {'sport', 'business', 'tech'}


# 8. evaluate — accuracy eşik üstü
def test_evaluate_accuracy(trained_pipeline, data):
    X, y = data
    acc = evaluate(trained_pipeline, X, y)
    assert 0.0 <= acc <= 1.0
    # Gerçek değer 1.0 — küçük veri setinde TF-IDF + LR mükemmel ayırır
    assert acc > 0.9


# 9. extract_entities — en az 1 varlık + label string
def test_extract_entities(data):
    ents = extract_entities(
        "Apple unveiled a new product in California while Serena Williams played in London."
    )
    assert isinstance(ents, list)
    assert len(ents) >= 1
    for text, label in ents:
        assert isinstance(text, str) and text
        assert isinstance(label, str) and label


# 10. semantic_search — top_k + azalan skor + ilgili ilk
def test_semantic_search(data):
    X, _ = data
    results = semantic_search("a player scored a goal in the match", X, top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    # (doc, score) tuple yapısı
    for doc, score in results:
        assert isinstance(doc, str)
        assert isinstance(score, float)
    # Skorlar azalan sırada
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # En ilgili doküman bir spor metni olmalı (goal/match/scored/penalty/player...)
    top_doc = results[0][0].lower()
    assert any(w in top_doc for w in
               ['goal', 'match', 'scored', 'penalty', 'player', 'game', 'team'])


# 11. semantic_search — top_k sınırı
def test_semantic_search_topk(data):
    X, _ = data
    results = semantic_search("interest rates and profits", X, top_k=2)
    assert len(results) == 2


# 12. save_model / load_model — round-trip
def test_save_load_roundtrip(trained_pipeline, tmp_path):
    path = os.path.join(str(tmp_path), "model.joblib")
    save_model(trained_pipeline, path)
    assert os.path.exists(path)
    loaded = load_model(path)
    sample = "The stock market fell amid inflation fears."
    # Yüklenen model aynı tahmini vermeli
    assert classify(loaded, sample) == classify(trained_pipeline, sample)


# 13. run_pipeline — uçtan uca
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'accuracy', 'sample_prediction', 'entities', 'search_results'
    }
    assert result['accuracy'] > 0.9
    assert isinstance(result['sample_prediction'], str)
    assert len(result['entities']) >= 1
    assert len(result['search_results']) >= 1


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
