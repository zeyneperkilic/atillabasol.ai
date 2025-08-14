#!/bin/bash

echo "🚀 AB AI MODEL - Tek Seferde Kurulum Başlıyor..."
echo "=================================================="
echo ""
echo "📋 Bu script otomatik olarak:"
echo "   • Homebrew paket yöneticisini kurar"
echo "   • Java 17'yi kurar"
echo "   • Python 3.12'yi kurar"
echo "   • Tesseract (OCR) kurar"
echo "   • Virtual Environment oluşturur"
echo "   • Tüm Python paketlerini kurar"
echo ""
echo "⏱️  Tahmini süre: 15-30 dakika"
echo "🌐 İnternet bağlantısı gerekli"
echo ""

# Kullanıcıdan onay al
read -p "Kuruluma başlamak istiyor musunuz? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Kurulum iptal edildi."
    exit 1
fi

echo ""
echo "🚀 Kurulum başlıyor..."
echo "=================================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Homebrew kurulumu
echo -e "${BLUE}🍺 Homebrew kuruluyor...${NC}"
if ! command -v brew &> /dev/null; then
    echo "Homebrew bulunamadı, kuruluyor..."
    echo "⚠️  Bu işlem birkaç dakika sürebilir..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Homebrew PATH'i ekle
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    echo -e "${GREEN}✅ Homebrew kuruldu!${NC}"
else
    echo -e "${GREEN}✅ Homebrew zaten kurulu: $(brew --version | head -n 1)${NC}"
fi

# Java kurulumu (Homebrew ile)
echo -e "${BLUE}📦 Java kuruluyor...${NC}"
if ! command -v java &> /dev/null; then
    echo "Java bulunamadı, Homebrew ile kuruluyor..."
    echo "⚠️  Bu işlem birkaç dakika sürebilir..."
    brew install openjdk@17
    sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
    echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@17' >> ~/.zshrc
    echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
    source ~/.zshrc
    echo -e "${GREEN}✅ Java kuruldu!${NC}"
else
    echo -e "${GREEN}✅ Java zaten kurulu: $(java -version 2>&1 | head -n 1)${NC}"
fi

# Python kontrolü
echo -e "${BLUE}🐍 Python kontrol ediliyor...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Python3 bulunamadı, Homebrew ile kuruluyor..."
    echo "⚠️  Bu işlem birkaç dakika sürebilir..."
    brew install python@3.12
    echo -e "${GREEN}✅ Python3 kuruldu!${NC}"
else
    echo -e "${GREEN}✅ Python3 kurulu: $(python3 --version)${NC}"
fi

# Tesseract kurulumu (Homebrew ile)
echo -e "${BLUE}🔍 Tesseract (OCR) kuruluyor...${NC}"
if ! command -v tesseract &> /dev/null; then
    echo "Tesseract bulunamadı, Homebrew ile kuruluyor..."
    echo "⚠️  Bu işlem birkaç dakika sürebilir..."
    brew install tesseract tesseract-lang
    echo -e "${GREEN}✅ Tesseract kuruldu!${NC}"
else
    echo -e "${GREEN}✅ Tesseract zaten kurulu: $(tesseract --version | head -n 1)${NC}"
fi

# Virtual environment oluşturma
echo -e "${BLUE}🌐 Virtual Environment oluşturuluyor...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual Environment oluşturuldu!${NC}"
else
    echo -e "${GREEN}✅ Virtual Environment zaten mevcut!${NC}"
fi

# Virtual environment'ı aktifleştirme
echo -e "${BLUE}🔧 Virtual Environment aktifleştiriliyor...${NC}"
source venv/bin/activate

# Python paketlerini kurma
echo -e "${BLUE}📚 Python paketleri kuruluyor...${NC}"
if [ -f "requirements.txt" ]; then
    echo "⚠️  Bu işlem 10-15 dakika sürebilir..."
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Tüm paketler kuruldu!${NC}"
else
    echo -e "${RED}❌ requirements.txt bulunamadı!${NC}"
    exit 1
fi

# Kurulum tamamlandı
echo ""
echo -e "${GREEN}🎉 Kurulum Tamamlandı!${NC}"
echo "=================================================="
echo -e "${YELLOW}🚀 Uygulamayı başlatmak için:${NC}"
echo ""
echo "1. Virtual Environment'ı aktifleştirin:"
echo "   source venv/bin/activate"
echo ""
echo "2. Uygulamayı başlatın:"
echo "   python main.py"
echo ""
echo "3. Tarayıcıda açın:"
echo -e "${BLUE}   http://127.0.0.1:8000${NC}"
echo ""
echo "=================================================="
echo -e "${GREEN}🎯 Artık AI asistanınızı kullanmaya başlayabilirsiniz!${NC}"
echo ""

# Shell'i yeniden yükle
echo "🔄 Shell yeniden yükleniyor..."
exec zsh 