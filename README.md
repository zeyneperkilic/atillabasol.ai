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

## 🚀 Tek Seferde Kurulum

### **1. Scripti Çalıştırılabilir Yapın**
```bash
chmod +x install.sh
```

### **2. Kurulumu Başlatın**
```bash
./install.sh
```

**Bu script otomatik olarak:**
- 🍺 Homebrew kurar (yoksa)
- 📦 Java 17 kurar
- 🐍 Python 3.12 kurar
- 🔍 Tesseract (OCR) kurar
- 🌐 Virtual Environment oluşturur
- 📚 Tüm Python paketlerini kurar

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
cd /path/to/OPEN_ROUTER_AI_BACKUP_20250811_2315
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

## 📞 Destek

Herhangi bir sorun yaşarsanız:
1. Terminal çıktısını kontrol edin
2. Gerekli paketlerin kurulu olduğundan emin olun
3. Virtual environment'ın aktif olduğunu kontrol edin

---

**🎉 Kurulum tamamlandıktan sonra AI asistanınızı kullanmaya başlayabilirsiniz!**
