# DS-45 — Uçtan Uca NLP Pipeline (Capstone)

**Modül**: İleri NLP (Capstone) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir haber/medya platformunda **NLP mühendisi** olarak çalışıyorsun. Platforma gelen kısa metinleri tek bir **uçtan uca NLP pipeline**'ından geçirmen gerekiyor:

1. Metni **kategorilere** ayır (sport / business / tech),
2. İçindeki **kişi, kurum ve yer isimlerini** (named entity) çıkar,
3. Bir sorguya **anlamca en yakın** dokümanları **semantik arama** ile bul,
4. Eğittiğin modeli diske **kaydet ve geri yükle** (production'a çıkar).

Bu capstone, İleri NLP dersinde öğrendiğin **tüm araçları TEK pipeline'da** birleştirir. Tek tek küçük problemler değil; gerçek bir NLP sisteminin uçtan uca iskeleti.

## 🧩 Birleştirdiğin Araçlar

- ✅ **Metin ön işleme** — lowercase + regex temizliği + stopword eleme
- ✅ **TF-IDF + LogisticRegression** — sklearn `Pipeline` ile metin sınıflandırma
- ✅ **spaCy NER** (`en_core_web_sm`) — metinden varlık (ORG / PERSON / GPE...) çıkarımı
- ✅ **sentence-transformers** (`all-MiniLM-L6-v2`) — cümle embedding'i + **cosine** ile semantik arama
- ✅ **Model persistence** — `joblib` ile kaydet/yükle (round-trip)
- ✅ **Uçtan uca pipeline** — hepsini tek akışta birleştirme

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd ds-45-nlp-capstone

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# spaCy İngilizce modeli (NER için gerekli — bir kez)
python -m spacy download en_core_web_sm

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

> **Not — ilk çalıştırma:** `sentence-transformers` modeli (`all-MiniLM-L6-v2`) ilk kullanımda internetten indirilir (~90MB). spaCy modeli de bir kez indirilir. Sonraki çalıştırmalar lokal cache'ten okur.

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 725 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Veri Seti — Koda Gömülü

Bu projede dış veri **indirmiyorsun**. Küçük (~30-40 metin) İngilizce bir veri setini **`load_data()` fonksiyonunda koda gömeceksin**. 3-4 kategori (örn. `sport`, `business`, `tech`), her kategoride en az 2 (önerilen ~10-12) örnek cümle.

```python
X, y = load_data()
# X: ["The football team won the championship final...", ...]
# y: ["sport", "business", "tech", ...]
```

Capstone'un odağı veri toplamak değil; **araçları birleştirmek**. O yüzden veri küçük ve deterministik.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki fonksiyonları sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`load_data()`** — ~30-40 İngilizce metin + 3-4 kategori etiketi döndür (gömülü)
2. **`preprocess(text)`** — lowercase + regex + stopword temizliği
3. **`build_classifier()`** — `TfidfVectorizer` + `LogisticRegression` Pipeline
4. **`train_classifier(pipe, X, y)`** — fit + dön
5. **`classify(pipe, text)`** — tek metin için kategori (str)
6. **`evaluate(pipe, X, y)`** — accuracy (float)
7. **`extract_entities(text)`** — spaCy `en_core_web_sm` ile `[(text, label), ...]`
8. **`semantic_search(query, docs, top_k)`** — `all-MiniLM-L6-v2` + cosine, skor azalan `[(doc, score), ...]`
9. **`save_model(pipe, path)`** / **`load_model(path)`** — joblib round-trip
10. **`run_pipeline()`** — hepsini uçtan uca birleştir, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] Metni **ön işleyebilecek** (regex + stopword) ve **TF-IDF + LogisticRegression** ile sınıflandırabileceksin
- [x] **spaCy** ile metinden **named entity** çıkarabileceksin
- [x] **sentence-transformers** ile **embedding** üretip **cosine similarity** ile semantik arama yapabileceksin
- [x] Bir modeli **joblib** ile diske **kaydedip geri yükleyebileceksin**
- [x] Tüm bu araçları **tek bir uçtan uca pipeline**'da birleştirebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (13 test)

Tümü pass olmalı:
- `load_data` ~30-40 metin, 3-4 kategori, her kategoride ≥2 örnek
- `preprocess` küçük harf + rakam/noktalama temizliği + stopword eleme
- `build_classifier` doğru tip (**TfidfVectorizer** + **LogisticRegression**)
- `classify` bilinen bir kategori döndürür
- `evaluate` accuracy eşiği geçer (**> 0.9** — küçük veri setinde TF-IDF + LR mükemmele yakın ayırır)
- `extract_entities` en az 1 varlık döndürür ve label bir string
- `semantic_search` `top_k` sonuç döndürür, skorlar **azalan**, en ilgili doküman ilk sırada
- `save_model → load_model` sonrası model **aynı** tahmini verir
- `run_pipeline` tüm aşamaları çalıştırır, beklenen anahtarları döndürür

## 📊 Beklenen Sonuçlar

```
Accuracy: 1.0 (küçük, ayrışabilir veri setinde TF-IDF + LogisticRegression)
NER: ORG / PERSON / GPE gibi etiketlerle en az birkaç varlık
Semantik arama: spor sorgusunda en üst sonuç bir spor metni
Round-trip: kaydedilen ve yüklenen model aynı tahmini verir
```

## 💡 İpuçları

- spaCy ve sentence-transformers modelleri **büyüktür**; her çağrıda yeniden yükleme yapma — modülde **bir kez yükleyip global cache'le** (çözümdeki gibi).
- `preprocess`'i `TfidfVectorizer(preprocessor=preprocess)` olarak pipeline'a verebilirsin; böylece eğitim ve tahmin aynı temizlikten geçer.
- Embedding'leri `normalize_embeddings=True` ile üretirsen **cosine = dot product** olur (`doc_emb @ q_emb`).
- `np.argsort(scores)[::-1][:top_k]` → en büyük `top_k` skorun indeksleri (azalan).
- `random_state=42` değerini değiştirme.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
