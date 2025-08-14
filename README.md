# 🚀 AB AI MODEL - Tek Seferde Kurulum

Bu proje, çoklu AI model desteği ile çalışan gelişmiş bir AI asistan uygulamasıdır.

## ✨ Özellikler

- 🤖 **5 Ana AI Model Desteği:**
  - GPT-4o (OpenAI)
  - Gemini 2.5 Pro (Google)
  - Grok-4 (xAI)
  - Claude Sonnet 4 (Anthropic)
  - DeepSeek Chat (DeepSeek)

- 🌐 **Web Search Özelliği** (:online modeller)
- 📁 **Dosya Yükleme** (PDF, DOCX, TXT, JPG, PNG, MP4, AVI, MOV)
- 🔍 **OCR** (görsel metin okuma)
- 🎥 **Video İşleme** (frame extraction + OCR)
- 💾 **Sohbet Geçmişi ve Hafıza Sistemi**

## 🚀 Tam Kurulum Rehberi

### **1. Projeyi İndirin**
```bash
# GitHub'dan projeyi klonlayın
git clone https://github.com/zeyneperkilic/atillabasol.ai.git

# Proje klasörüne gidin
cd atillabasol.ai
```

**Veya ZIP olarak indirin:**
- GitHub'da "Code" butonuna tıklayın
- "Download ZIP" seçin
- ZIP'i açın ve klasöre gidin

### **2. Scripti Çalıştırılabilir Yapın**
```bash
# Terminal'i açın
# Proje klasörüne gidin
cd /path/to/atillabasol.ai

# Scripti çalıştırılabilir yapın
chmod +x install.sh
```

### **3. Kurulumu Başlatın**
```bash
# Kurulum scriptini çalıştırın
./install.sh
```

**Bu script otomatik olarak:**
- 🍺 Homebrew kurar (yoksa)
- 📦 Java 17 kurar
- 🐍 Python 3.12 kurar
- 🔍 Tesseract (OCR) kurar
- 🌐 Virtual Environment oluşturur
- 📚 Tüm Python paketlerini kurar

### **4. Kurulum Tamamlandıktan Sonra**
```bash
# Virtual environment'ı aktifleştirin
source venv/bin/activate

# Uygulamayı başlatın
python main.py
```

### **5. Tarayıcıda Açın**
```
http://127.0.0.1:8000
```

## 🔧 Manuel Kurulum (İsteğe Bağlı)

### **Gereksinimler:**
- macOS 10.15+
- Terminal erişimi
- İnternet bağlantısı

### **Adım Adım:**
```bash
# 1. Homebrew kurulumu
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Java kurulumu
brew install openjdk@17

# 3. Python kurulumu
brew install python@3.12

# 4. Tesseract kurulumu
brew install tesseract tesseract-lang

# 5. Proje kurulumu
cd /path/to/atillabasol.ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Uygulamayı Başlatma

```bash
# Virtual environment'ı aktifleştirin
source venv/bin/activate

# Uygulamayı başlatın
python main.py
```

## 🌐 Erişim

Uygulama başladıktan sonra tarayıcınızda şu adresi açın:
```
http://127.0.0.1:8000
```

## ⚙️ Konfigürasyon

`.env` dosyasında API anahtarınızı ayarlayın:
```env
OPENROUTER_API_KEY=your_api_key_here
APP_REFERER=http://localhost:8000
APP_TITLE=AB AI MODEL
```

## 🆘 Sorun Giderme

### **Java Hatası:**
```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH=$JAVA_HOME/bin:$PATH
```

### **Tesseract Hatası:**
```bash
brew install tesseract tesseract-lang
```

### **Python Hatası:**
```bash
brew install python@3.12
```

### **Permission Hatası:**
```bash
chmod +x install.sh
```

### **Git Hatası:**
```bash
# Git yoksa Homebrew ile kurun
brew install git
```

## 📱 Kurulum Sonrası

### **İlk Kullanım:**
1. Uygulamayı başlatın: `python main.py`
2. Tarayıcıda `http://127.0.0.1:8000` açın
3. API anahtarınızı `.env` dosyasında ayarlayın
4. AI modellerini seçin ve kullanmaya başlayın!

### **Güncellemeler:**
```bash
# Projeyi güncelleyin
git pull origin main

# Paketleri güncelleyin
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## 📞 Destek

Herhangi bir sorun yaşarsanız:
1. Terminal çıktısını kontrol edin
2. Gerekli paketlerin kurulu olduğundan emin olun
3. Virtual environment'ın aktif olduğunu kontrol edin
4. README'deki sorun giderme bölümünü kontrol edin

---

**🎉 Kurulum tamamlandıktan sonra AI asistanınızı kullanmaya başlayabilirsiniz!**

**💡 İpucu:** Kurulum sırasında herhangi bir hata alırsanız, script otomatik olarak size yardımcı olacaktır.
