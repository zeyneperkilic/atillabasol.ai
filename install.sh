#!/bin/bash

echo "🚀 ATILLA BASOL AI MODEL - TAM OTOMATİK KURULUM"
echo "=================================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python kontrolü ve kurulumu
echo -e "${BLUE}🔍 Python kontrol ediliyor...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı! Kuruluyor...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo -e "${YELLOW}📦 Homebrew ile Python3 kuruluyor...${NC}"
            brew install python3
        else
            echo -e "${YELLOW}📥 Homebrew kuruluyor...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            echo -e "${YELLOW}📦 Python3 kuruluyor...${NC}"
            brew install python3
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "${YELLOW}📦 Linux için Python3 kuruluyor...${NC}"
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv
    else
        echo -e "${RED}❌ Desteklenmeyen işletim sistemi: $OSTYPE${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Python3 bulundu: $(python3 --version)${NC}"

# Gerekli klasörleri oluştur
echo -e "${BLUE}📁 Gerekli klasörler oluşturuluyor...${NC}"
mkdir -p chats uploads static templates

# Virtual environment oluştur
echo -e "${BLUE}📦 Virtual environment oluşturuluyor...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Eski venv siliniyor...${NC}"
    rm -rf venv
fi
python3 -m venv venv

# Virtual environment aktif et
echo -e "${BLUE}🔧 Virtual environment aktif ediliyor...${NC}"
source venv/bin/activate

# Pip güncelle
echo -e "${BLUE}⬆️  Pip güncelleniyor...${NC}"
pip install --upgrade pip

# Gerekli kütüphaneleri kur
echo -e "${BLUE}📚 Gerekli kütüphaneler kuruluyor...${NC}"
pip install fastapi uvicorn jinja2 python-multipart python-dotenv httpx requests PyPDF2 python-docx pillow moviepy opencv-python itsdangerous

# OCR kütüphaneleri (opsiyonel)
echo -e "${BLUE}🔍 OCR kütüphaneleri kuruluyor (opsiyonel)...${NC}"
pip install pytesseract

# Tesseract kurulumu (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}🖼️  macOS için Tesseract kurulumu (opsiyonel)...${NC}"
    if command -v brew &> /dev/null; then
        echo -e "${GREEN}✅ Homebrew bulundu, Tesseract kuruluyor...${NC}"
        if brew install tesseract tesseract-lang; then
            echo -e "${GREEN}✅ Tesseract başarıyla kuruldu!${NC}"
        else
            echo -e "${YELLOW}⚠️  Tesseract kurulumu başarısız, OCR olmadan devam ediliyor...${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Homebrew bulunamadı${NC}"
        echo -e "${BLUE}📥 Tesseract manuel kurulum için:${NC}"
        echo "1. https://github.com/UB-Mannheim/tesseract/wiki adresinden indirin"
        echo "2. .pkg dosyasını çift tıklayarak kurun"
        echo ""
        echo -e "${BLUE}🔧 Veya Homebrew kurmak için:${NC}"
        echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo -e "${YELLOW}⏭️  Şimdilik Tesseract olmadan devam ediliyor...${NC}"
        echo -e "${YELLOW}⚠️  OCR özelliği çalışmayacak ama uygulama çalışacak!${NC}"
    fi
fi

# .env dosyası oluştur
echo -e "${BLUE}🔑 .env dosyası oluşturuluyor...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << EOF
OPENROUTER_API_KEY=your_api_key_here
APP_REFERER=http://localhost:8000
APP_TITLE=ATILLA BASOL AI MODEL
EOF
    echo -e "${YELLOW}⚠️  .env dosyası oluşturuldu. Lütfen OPENROUTER_API_KEY'i güncelleyin!${NC}"
else
    echo -e "${GREEN}✅ .env dosyası zaten mevcut${NC}"
fi

# Uygulamayı başlat
echo ""
echo -e "${GREEN}🎉 Kurulum tamamlandı!${NC}"
echo ""
echo -e "${BLUE}🚀 Uygulama başlatılıyor...${NC}"
echo -e "${YELLOW}⚠️  Lütfen .env dosyasında API key'i güncelleyin!${NC}"
echo ""

# Otomatik başlatma
echo -e "${BLUE}⏳ 5 saniye sonra uygulama başlayacak...${NC}"
echo -e "${BLUE}📱 Tarayıcı otomatik açılacak...${NC}"
echo ""

sleep 5

# Uygulamayı başlat ve tarayıcıyı aç
echo -e "${GREEN}🚀 Uygulama başlatılıyor...${NC}"
echo -e "${BLUE}🌐 Tarayıcı açılıyor...${NC}"

# Arka planda uygulamayı başlat
python3 main.py &
APP_PID=$!

# 3 saniye bekle
sleep 3

# Tarayıcıyı aç
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://127.0.0.1:8000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open http://127.0.0.1:8000
    elif command -v gnome-open &> /dev/null; then
        gnome-open http://127.0.0.1:8000
    else
        echo -e "${YELLOW}⚠️  Tarayıcı otomatik açılamadı. Manuel olarak http://127.0.0.1:8000 adresine gidin${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Uygulama başlatıldı!${NC}"
echo -e "${BLUE}🌐 Tarayıcıda http://127.0.0.1:8000 adresine gidin${NC}"
echo -e "${BLUE}🔄 Uygulamayı durdurmak için: kill $APP_PID${NC}"
echo ""
echo -e "${BLUE}🎯 Uygulama özellikleri:${NC}"
echo "• Çoklu AI model desteği (GPT-4o, Claude, Gemini, Grok)"
echo "• Web search özelliği (:online modeller)"
echo "• Dosya yükleme (PDF, DOCX, TXT, JPG, PNG, MP4, AVI, MOV)"
echo "• OCR (görsel metin okuma)"
echo "• Video işleme (frame extraction + OCR)"
echo "• Sohbet geçmişi ve hafıza sistemi"
echo ""
echo -e "${GREEN}🚀 İyi kullanımlar!${NC}"

# Uygulama çalışırken bekle
wait $APP_PID 