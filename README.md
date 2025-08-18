# AB AI MODEL - OpenRouter AI Asistanı

## 🚀 Hızlı Başlangıç

### Start/Stop Icon Sistemi (Önerilen)

**Terminal ile uğraşmadan projeyi başlatmak/durdurmak için:**

#### 🟢 **Başlatmak:**
**Seçenek 1: Görsel Icon'a çift tıkla**
- `start_ai_shortcut.command` dosyasına **çift tıklayın**
- Terminal otomatik açılır ve proje başlar
- Tarayıcı otomatik açılır

**Seçenek 2: Eski yöntem**
- `start_ai.command` dosyasına **çift tıklayın**

**Seçenek 3: Terminal komutu**
```bash
./start_ai.sh
```

#### 🔴 **Durdurmak:**
**Seçenek 1: Görsel Icon'a çift tıkla**
- `stop_ai_shortcut.command` dosyasına **çift tıklayın**
- Terminal otomatik açılır ve proje durur

**Seçenek 2: Eski yöntem**
- `stop_ai.command` dosyasına **çift tıklayın**

**Seçenek 3: Terminal komutu**
```bash
./stop_ai.sh
```

### Manuel Kurulum (Eski Yöntem)

### 1. Homebrew Kurulumu
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Gerekli Paketleri Kurun
```bash
brew install openjdk@17 python@3.12 tesseract tesseract-lang
```

### 3. Python Bağımlılıkları
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. API Key Ayarlayın
`.env` dosyası oluşturun:
```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

## 📱 Kullanım

### 🌟 Ana Özellikler
- **Cover Page**: Giriş sayfası
- **Main Page**: AI modellerle sohbet
- **Sol Menü**: Sohbet geçmişi ve ayarlar
- **Model Seçimi**: Üstte aktif modeller, altta ekstra modeller

### 🔄 Sohbet Özellikleri
- Yeni sohbet başlatma
- Sohbet geçmişini görüntüleme
- Sohbet silme
- Dosya yükleme ve analiz

## 🏗️ Teknik Detaylar

### 🐍 Teknolojiler
- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Veritabanı**: SQLite
- **OCR**: Tesseract
- **Dosya İşleme**: PyPDF2, python-docx, PIL

### 📁 Proje Yapısı
```
OPEN_ROUTER_AI/
├── main.py              # Ana uygulama
├── install.sh           # Otomatik kurulum script'i
├── requirements.txt     # Python bağımlılıkları
├── .env                # API Key (otomatik oluşturulur)
├── templates/          # HTML şablonları
├── static/             # CSS, JS, resimler
├── uploads/            # Yüklenen dosyalar
├── memory/             # Sohbet geçmişi
└── chats/              # Sohbet verileri
```

## 🚨 Sorun Giderme

### Port 8000 Kullanımda
```bash
# Çalışan process'leri bulun
lsof -ti:8000

# Process'i sonlandırın
kill -9 <PID>
```

### Virtual Environment Hatası
```bash
# Venv'i yeniden oluşturun
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API Key Hatası
```bash
# .env dosyasını kontrol edin
cat .env

# Yeniden oluşturun
echo "OPENROUTER_API_KEY=your_new_api_key" > .env
```

## 📞 Destek

- **GitHub Issues**: [Proje sayfasında](https://github.com/your-repo/issues)
- **Dokümantasyon**: Bu README dosyası
- **API Key**: [OpenRouter Keys](https://openrouter.ai/keys)

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

## 🎉 Kurulum Tamamlandı!

Artık AI asistanınızı kullanmaya başlayabilirsiniz! 

**🚀 Tek komutla kurulum: `./install.sh`**
