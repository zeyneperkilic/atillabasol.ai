#!/bin/bash

echo "🚀 ATILLA BASOL AI MODEL Kurulum Scripti"
echo "=========================================="

# Python kontrolü
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı! Lütfen Python3'ü kurun."
    echo "macOS: brew install python3"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "✅ Python3 bulundu: $(python3 --version)"

# Virtual environment oluştur
echo "📦 Virtual environment oluşturuluyor..."
python3 -m venv venv

# Virtual environment aktif et
echo "🔧 Virtual environment aktif ediliyor..."
source venv/bin/activate

# Pip güncelle
echo "⬆️ Pip güncelleniyor..."
pip install --upgrade pip

# Gerekli kütüphaneleri kur
echo "📚 Gerekli kütüphaneler kuruluyor..."
pip install fastapi uvicorn jinja2 aiofiles python-multipart python-dotenv httpx requests PyPDF2 python-docx pillow pytesseract moviepy opencv-python

# Tesseract kurulumu (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖼️ macOS için Tesseract kuruluyor..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew bulunamadı! Lütfen Homebrew kurun: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    brew install tesseract tesseract-lang
fi

# .env dosyası oluştur
echo "🔑 .env dosyası oluşturuluyor..."
cat > .env << EOF
OPENROUTER_API_KEY=your_api_key_here
APP_REFERER=http://localhost:8000
APP_TITLE=ATILLA BASOL AI MODEL
EOF

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "📝 Yapmanız gerekenler:"
echo "1. .env dosyasında OPENROUTER_API_KEY'i güncelleyin"
echo "2. Uygulamayı başlatmak için: source venv/bin/activate && python main.py"
echo "3. Tarayıcıda http://127.0.0.1:8000 adresine gidin"
echo ""
echo "🎯 Uygulama özellikleri:"
echo "- Çoklu AI model desteği (GPT-4o, Claude, Gemini, Grok)"
echo "- Web search özelliği (:online modeller)"
echo "- Dosya yükleme (PDF, DOCX, TXT, JPG, PNG, MP4, AVI, MOV)"
echo "- OCR (görsel metin okuma)"
echo "- Video işleme (frame extraction + OCR)"
echo "- Sohbet geçmişi ve hafıza sistemi"
echo ""
echo "🚀 İyi kullanımlar!" 