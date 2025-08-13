#!/bin/bash

MAX_RETRY=3
RETRY_COUNT=0

start_services() {
    echo ">>> ESKİ SÜREÇLER TEMİZLENİYOR..."
    pkill -f "ollama" 2>/dev/null
    pkill -f "uvicorn" 2>/dev/null
    pkill -f "python.*main:app" 2>/dev/null

    for PORT in 11434 8000; do
        if lsof -i tcp:$PORT >/dev/null 2>&1; then
            echo ">>> PORT $PORT KULLANILIYOR, TEMİZLENİYOR..."
            lsof -ti tcp:$PORT | xargs kill -9 2>/dev/null
        fi
    done
    echo ">>> TÜM SÜREÇLER VE PORTLAR TEMİZLENDİ!"

    echo ">>> OLLAMA BAŞLATILIYOR..."
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!

    echo ">>> OLLAMA PORT HAZIR OLMASI BEKLENİYOR..."
    until nc -z 127.0.0.1 11434; do
        sleep 1
    done
    echo ">>> OLLAMA PORT HAZIR!"

    echo ">>> MODELLERİN HAZIR OLMASI BEKLENİYOR..."
    sleep 5
    echo ">>> MODELLER HAZIR!"

    cd ~/Desktop/AtillaBasolAI/backend || exit 1
    if [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
        echo ">>> FASTAPI UVICORN BAŞLATILIYOR..."
        uvicorn main:app --reload > /tmp/uvicorn.log 2>&1 &
        UVICORN_PID=$!
    else
        echo ">>> HATA: Sanal ortam (venv) bulunamadı!"
        exit 1
    fi

    echo ">>> UVICORN PORT HAZIR OLMASI BEKLENİYOR..."
    until nc -z 127.0.0.1 8000; do
        sleep 1
    done
    echo ">>> UVICORN HAZIR!"
}

health_check() {
    echo ">>> SAĞLIK KONTROLÜ YAPILIYOR..."
    OLLAMA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434)
    UVICORN_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000)

    if [[ "$OLLAMA_STATUS" =~ ^(200|404)$ ]]; then
        echo "✔ Ollama yanıt veriyor (HTTP $OLLAMA_STATUS)"
    else
        echo "✖ Ollama yanıt vermiyor (HTTP $OLLAMA_STATUS)"
        return 1
    fi

    if [[ "$UVICORN_STATUS" =~ ^(200|404)$ ]]; then
        echo "✔ Uvicorn yanıt veriyor (HTTP $UVICORN_STATUS)"
    else
        echo "✖ Uvicorn yanıt vermiyor (HTTP $UVICORN_STATUS)"
        return 1
    fi

    return 0
}

while [ $RETRY_COUNT -lt $MAX_RETRY ]; do
    start_services
    if health_check; then
        echo ">>> SİSTEM BAŞARILI BİR ŞEKİLDE BAŞLATILDI! (Ollama PID: $OLLAMA_PID, Uvicorn PID: $UVICORN_PID)"
        open http://127.0.0.1:8000
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo ">>> Sağlık kontrolü başarısız! ($RETRY_COUNT/$MAX_RETRY)"
        if [ $RETRY_COUNT -lt $MAX_RETRY ]; then
            echo ">>> Sistem yeniden başlatılıyor..."
            sleep 3
        fi
    fi
done

echo "✖ SİSTEM $MAX_RETRY DENEMEDEN SONRA BAŞARILI BAŞLATILAMADI!"
exit 1

