import asyncio
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets
import httpx
import time
import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
import re
from typing import List, Optional

import shutil
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# SQLite Database setup
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Database bağlantısı için context manager"""
    conn = sqlite3.connect('ai_chat.db')
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Database tablolarını oluştur"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Conversations tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            model TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Global memory tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        print("DEBUG: Database tabloları oluşturuldu")

# Database'i başlat
init_database()

# === CONFIG ===
# OpenRouter chat-completions endpoint (OpenAI uyumlu)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
APP_REFERER = os.getenv("APP_REFERER", "http://localhost:8000")
APP_TITLE = os.getenv("APP_TITLE", "AtillaBasolAI")

if not OPENROUTER_API_KEY:
    print("UYARI: OPENROUTER_API_KEY ortam değişkeni tanımlı değil. Lütfen .env içine ekleyin.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Session middleware ekle
app.add_middleware(
    SessionMiddleware,
    secret_key=secrets.token_urlsafe(32),
    max_age=3600  # 1 saat
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

CHAT_DIR = "chats"
UPLOAD_DIR = "uploads"
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenRouter model slug'ları (Premium Models) - UI'da gösterilecek 5 model
MODELS = [
  "openai/gpt-4o:online", 
  "google/gemini-2.5-pro:online",
  "x-ai/grok-4:online",
  "anthropic/claude-sonnet-4:online",
  "deepseek/deepseek-chat-v3-0324:online",
]

# UI'da görünen açıklamalar (Premium Models)
MODEL_DESCRIPTIONS = {
    "openai/gpt-4o:online": "Coding, general problem-solving, creative writing",
    "google/gemini-2.5-pro:online": "Research, multimodal tasks, fact-checking",
    "x-ai/grok-4:online": "Casual conversation, humor, creative brainstorming",
    "anthropic/claude-sonnet-4:online": "Long-form writing, safe reasoning, detailed analysis",
    "deepseek/deepseek-chat-v3-0324:online": "Technical explanations, data analysis, coding",
}

# Deneyebileceğiniz Başka Modeller (Open Source & Specialized)
AVAILABLE_MODELS = {
    "openai/gpt-4o:online": "Coding, general problem-solving, creative writing",
    "google/gemini-2.5-pro:online": "Research, multimodal tasks, fact-checking",
    "x-ai/grok-4:online": "Casual conversation, humor, creative brainstorming",
    "anthropic/claude-sonnet-4:online": "Long-form writing, safe reasoning, detailed analysis",
    "deepseek/deepseek-chat-v3-0324:online": "Technical explanations, data analysis, coding",
    
    # Open Source Models
    "meta-llama/llama-3.3-70b-instruct:free": "LLAMA 3.3 70B INSTRUCT:FREE (High reasoning, detailed answers, versatile general use)",
    "mistralai/mistral-small-3.2-24b-instruct:free": "MISTRAL SMALL 3.2 24B INSTRUCT:FREE (Fast, efficient, good for short tasks and summaries)",
    "nousresearch/deephermes-3-llama-3-8b-preview:free": "DEEPHERMES 3 LLAMA 3 8B PREVIEW:FREE (Conversational tone, creative writing, roleplay)",
    "deepseek/deepseek-chat-v3-0324:free": "DEEPSEEK CHAT V3 0324:FREE (Technical Q&A, problem-solving, coding support)",
    "moonshotai/kimi-k2:free": "KIMI K2:FREE (Multilingual chat, general conversation, cultural knowledge)",
    "openai/gpt-oss-20b:free": "GPT OSS 20B:FREE (Balanced reasoning and creativity, versatile open-source option)",
    
}

# (Optional) "time sensitive" search snippet addition is available:
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CSE_ID = os.getenv("CSE_ID", "")

def google_search_sync(query):
    if not GOOGLE_API_KEY or not CSE_ID:
        return ""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_API_KEY, "cx": CSE_ID}
    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()
        snippets = [item.get("snippet","") for item in data.get("items", [])]
        return "\n".join(snippets[:5])
    except Exception:
        return ""

def detect_language(text):
    turkish_chars = "çğıöşü"
    return "tr" if any(c in text.lower() for c in turkish_chars) else "en"

def is_time_sensitive(question):
    keywords = ["current", "latest", "actual", "son", "şu an", "bugün", "kimdir", "şimdi", "halen", "now", "güncel"]
    q = question.lower()
    return any(k in q for k in keywords)

def save_chat_file(filename, content):
    with open(os.path.join(CHAT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def load_chats():
    chats = []
    chat_files = []
    
    # First collect all chat files
    for fname in os.listdir(CHAT_DIR):
        if fname.endswith(".json"):
            file_path = os.path.join(CHAT_DIR, fname)
            try:
                # Get file creation time
                file_time = os.path.getctime(file_path)
                chat_files.append((fname, file_time))
            except:
                # In case of error, extract timestamp from filename
                try:
                    # Try to extract timestamp from filename (e.g., "question_0813_1430.json")
                    timestamp_match = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})', fname)
                    if timestamp_match:
                        year, month, day, hour, minute = timestamp_match.groups()
                        file_time = time.mktime((int(year), int(month), int(day), int(hour), int(minute), 0, 0, 0, -1))
                    else:
                        file_time = 0
                    chat_files.append((fname, file_time))
                except:
                    file_time = 0
                    chat_files.append((fname, file_time))
    
    # Sort by timestamp (newest first)
    chat_files.sort(key=lambda x: x[1], reverse=True)
    
    # Load chats from sorted files
    for fname, _ in chat_files:
        try:
            with open(os.path.join(CHAT_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                data["filename"] = fname
                
                # Create title for old chats
                if "title" not in data:
                    import re
                    question = data.get("question", "")
                    clean_text = re.sub(r'[^\w\s]', '', question)
                    words = clean_text.split()[:4]
                    data["title"] = ' '.join(words) if words else "Old Chat"
                    if len(data["title"]) > 50:
                        data["title"] = data["title"][:47] + "..."
                
                chats.append(data)
        except Exception as e:
            print(f"DEBUG: Chat file loading error {fname}: {str(e)}")
            continue
    
    return chats

def delete_chat_file(filename):
    try:
        os.remove(os.path.join(CHAT_DIR, filename))
    except:
        pass

# -------- Conversation Memory System --------
MEMORY_DIR = "memory"  # Folder for memory files
GLOBAL_MEMORY_FILE = "global_memory.json"  # Single file for global memory
conversation_counter = 0

# Create memory folder
os.makedirs(MEMORY_DIR, exist_ok=True)

def generate_session_id():
    """Benzersiz session ID oluştur"""
    return f"session_{int(time.time())}_{secrets.token_hex(8)}"

def generate_conversation_id(session_id: str):
    """Session'a özel conversation ID oluştur"""
    global conversation_counter
    conversation_counter += 1
    return f"{session_id}_conv_{int(time.time())}_{conversation_counter}"

def save_memory_to_db(conversation_id: str, question: str, response: str, model: str = None):
    """Memory'yi database'e kaydet"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO conversations (conversation_id, question, response, model)
            VALUES (?, ?, ?, ?)
            ''', (conversation_id, question, response, model))
            conn.commit()
            print(f"DEBUG: Memory database'e kaydedildi: {conversation_id}")
    except Exception as e:
        print(f"DEBUG: Database save error: {str(e)}")

def load_memory_from_db(conversation_id: str, limit: int = 10) -> list:
    """Load memory from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT question, response, model, timestamp
            FROM conversations 
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (conversation_id, limit))
            
            rows = cursor.fetchall()
            memory_data = []
            for row in rows:
                memory_data.append({
                    'question': row[0],
                    'response': row[1],
                    'model': row[2],
                    'timestamp': row[3]
                })
            
            print(f"DEBUG: Memory loaded from database: {conversation_id}, {len(memory_data)} records")
            return memory_data
    except Exception as e:
        print(f"DEBUG: Database loading error: {str(e)}")
        return []

def load_global_memory() -> dict:
    """Load global memory from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM global_memory')
            rows = cursor.fetchall()
            
            global_data = {}
            for row in rows:
                global_data[row[0]] = row[1]
            
            return global_data
    except Exception as e:
        print(f"DEBUG: Global memory loading error: {str(e)}")
        return {}

def save_global_memory(key: str, value: str):
    """Global memory'yi database'e kaydet"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO global_memory (key, value)
            VALUES (?, ?)
            ''', (key, value))
            conn.commit()
            print(f"DEBUG: Global memory kaydedildi: {key}")
    except Exception as e:
        print(f"DEBUG: Global memory save error: {str(e)}")

def add_to_global_memory(key: str, value: str):
    """Global memory'ye bilgi ekle (isim, tercihler, vb.)"""
    save_global_memory(key, value)

def get_from_global_memory(key: str) -> str:
    """Global memory'den bilgi al"""
    global_data = load_global_memory()
    return global_data.get(key, "")

def add_to_memory(conversation_id: str, user_message: str, ai_responses: list, synthesis: str = ""):
    """Conversation memory'ye yeni mesaj ekle ve database'e kaydet"""
    # Her AI yanıtını database'e kaydet
    for response in ai_responses:
        if isinstance(response, dict) and 'text' in response:
            save_memory_to_db(conversation_id, user_message, response['text'], response.get('model', 'unknown'))
    
    # Sentez yanıtını da kaydet
    if synthesis:
        save_memory_to_db(conversation_id, user_message, synthesis, 'synthesis')

def get_conversation_context(conversation_id: str, max_messages: int = 5) -> str:
    """Conversation context'ini döndür (önceki mesajlar + global memory)"""
    # Local conversation memory
    memory_data = load_memory_from_db(conversation_id, max_messages)
    
    # Global memory (isim, tercihler, vb.)
    global_context = ""
    global_data = load_global_memory()
    
    if global_data:
        global_context = "🌍 General Information:\n"
        for key, value in global_data.items():
            global_context += f"• {key}: {value}\n"
        global_context += "---\n"
    
    # Local conversation context
    local_context = ""
    if memory_data:
        local_context = "📚 This Conversation History:\n"
        
        for entry in memory_data:
            local_context += f"👤 User: {entry['question']}\n"
            local_context += f"🤖 AI: {entry['response']}\n"
            local_context += "---\n"
    
    return global_context + local_context

# -------- File Processing Functions --------
async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    if not file.filename:
        return ""
    
    # Create safe filename
    safe_filename = f"{int(time.time())}_{file.filename}"
    file_path = Path(UPLOAD_DIR) / safe_filename
    
    # Save file
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return str(file_path)

def extract_text_from_file(file_path: str) -> str:
    """Extract text from file"""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                return f"PDF reading requires PyPDF2 library. File: {Path(file_path).name}"
        
        elif file_ext in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                return f"Word documents require python-docx library. File: {Path(file_path).name}"
        
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            try:
                from PIL import Image
                import pytesseract
                import subprocess
                import shutil
                
                # Tesseract'ın kurulu olup olmadığını kontrol et
                tesseract_path = shutil.which('tesseract')
                if not tesseract_path:
                    # Render'da farklı path'leri dene
                    possible_paths = [
                        '/usr/bin/tesseract',
                        '/usr/local/bin/tesseract',
                        '/opt/homebrew/bin/tesseract'
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            print(f"DEBUG: Tesseract found: {path}")
                            break
                    else:
                        return f"�� Image file: {Path(file_path).name}\nTesseract OCR engine not found. Installation required in Render."
                
                # Görseli aç
                image = Image.open(file_path)
                
                # OCR ile metin çıkar
                text = pytesseract.image_to_string(image, lang='eng+tur')
                
                if text.strip():
                    return f"📸 Image Content (OCR):\n{text.strip()}"
                else:
                    return f"📸 Image file: {Path(file_path).name}\nText not detected (OCR result empty)"
                    
            except ImportError:
                return f"📸 Image file: {Path(file_path).name}\nOCR requires pytesseract library. Installation: pip install pytesseract"
            except Exception as e:
                return f"📸 Image file: {Path(file_path).name}\nOCR error: {str(e)}"
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
            try:
                print(f"DEBUG: Video processing started: {file_path}")
                try:
                    from moviepy import VideoFileClip
                    import cv2
                    import numpy as np
                    from PIL import Image
                    import pytesseract
                    print(f"DEBUG: Libraries imported successfully")
                except ImportError as import_error:
                    print(f"DEBUG: Import error: {str(import_error)}")
                    return f"🎬 Video file: {Path(file_path).name}\nVideo processing requires moviepy and opencv libraries. Installation: pip install moviepy opencv-python"
                
                # Open video file
                video = VideoFileClip(file_path)
                print(f"DEBUG: Video opened: {file_path}")
                
                # Video bilgileri
                duration = video.duration
                fps = video.fps
                total_frames = int(duration * fps)
                print(f"DEBUG: Video info - Duration: {duration}s, FPS: {fps}, Frame: {total_frames}")
                
                # Perform OCR on keyframes (every 2 seconds)
                frame_interval = max(1, int(fps * 2))  # Every 2 seconds
                extracted_texts = []
                
                for i in range(0, total_frames, frame_interval):
                    if i < total_frames:
                        try:
                            # Get frame
                            frame = video.get_frame(i / fps)
                            print(f"DEBUG: Frame {i//fps:.1f}s captured")
                            
                            # Convert to PIL Image
                            from PIL import Image
                            pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')
                            
                            # Perform OCR
                            text = pytesseract.image_to_string(pil_image, lang='eng+tur')
                            if text.strip():
                                extracted_texts.append(f"Frame {i//fps:.1f}s: {text.strip()}")
                                print(f"DEBUG: Text found in Frame {i//fps:.1f}s: {text.strip()[:50]}...")
                        except Exception as frame_error:
                            print(f"DEBUG: Frame {i//fps:.1f}s error: {str(frame_error)}")
                            continue
                
                video.close()
                print(f"DEBUG: Video closed, {len(extracted_texts)} frames processed")
                
                if extracted_texts:
                    return f"🎬 Video Content (OCR):\nDuration: {duration:.1f}s, FPS: {fps:.1f}\n\n" + "\n\n".join(extracted_texts[:10])  # First 10 frames
                else:
                    return f"🎬 Video file: {Path(file_path).name}\nDuration: {duration:.1f}s, FPS: {fps:.1f}\nText not detected"
                    
            except ImportError as import_error:
                print(f"DEBUG: Import error: {str(import_error)}")
                return f"🎬 Video file: {Path(file_path).name}\nVideo processing requires moviepy and opencv libraries. Installation: pip install moviepy opencv-python"
            except Exception as e:
                print(f"DEBUG: General video processing error: {str(e)}")
                return f"🎬 Video file: {Path(file_path).name}\nVideo processing error: {str(e)}"
        
        else:
            return f"Unsupported file type: {file_ext}. File: {Path(file_path).name}"
    
    except Exception as e:
        return f"File reading error: {str(e)}"

def cleanup_old_files():
    """Clean up uploaded files older than 1 hour"""
    try:
        current_time = time.time()
        for file_path in Path(UPLOAD_DIR).glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 3600:  # 1 saat
                    file_path.unlink()
    except Exception:
        pass

def cleanup_old_memory():
    """7 günden eski memory dosyalarını temizle"""
    try:
        current_time = time.time()
        for filename in os.listdir(MEMORY_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(MEMORY_DIR, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 604800:  # 7 gün
                        os.remove(file_path)
                        print(f"DEBUG: Old memory deleted: {filename}")
    except Exception as e:
        print(f"DEBUG: Memory cleanup error: {str(e)}")



# -------- OpenRouter yardımcıları --------
def _headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": APP_REFERER,
        "X-Title": APP_TITLE,
        "Content-Type": "application/json",
    }

def _messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _extract_text(data: dict) -> str:
    # OpenRouter (OpenAI format): choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: bazı sağlayıcılar farklı döndürebilir
        return data.get("response", "").strip() or "ERROR: Could not get response."

# ----------------------- MODEL YANITI GETİRME -----------------------
async def fetch_model_answer(model: str, system_prompt: str, user_prompt: str):
    t0 = time.time()
    
    # Web search enabled models (only those with :online suffix)
    web_search_models = [
        "openai/gpt-4o:online",
        "google/gemini-2.5-pro:online",
        "x-ai/grok-4:online",
        "anthropic/claude-sonnet-4:online",
        "deepseek/deepseek-chat-v3-0324:online"
    ]
    
    payload = {
        "model": model,
        "messages": _messages(system_prompt, user_prompt),
        "stream": False,
        "temperature": 0.2,
    }
    
    # Add web_search parameter for web search enabled models
    if model in web_search_models:
        # Use OpenRouter web search via plugins parameter
        payload["plugins"] = [{"id": "web"}]
        print(f"DEBUG: Web search enabled: {model}")
        print(f"DEBUG: Web search payload: {payload}")
    
    # As per request: reasoning parameters are supported by models that support them,
    # but we keep them closed for universal compatibility.
    # "reasoning": {"effort": "medium"}

    attempt = 1
    text = "ERROR: Could not get response."
    while attempt <= 2:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(OPENROUTER_API_URL, headers=_headers(), json=payload)
                data = resp.json()
                text = _extract_text(data)
            break
        except Exception as e:
            if attempt == 2:
                text = f"ERROR: {e}"
        attempt += 1

    elapsed = time.time() - t0
    return {"model": model, "text": text, "elapsed": elapsed}

# ----------------------- COVER SAYFASI -----------------------
@app.get("/", response_class=HTMLResponse)
async def cover_page(request: Request):
    """Cover/Splash sayfası"""
    return templates.TemplateResponse("cover.html", {
        "request": request,
        "available_models": AVAILABLE_MODELS
    })

# ----------------------- ANA SAYFA -----------------------
@app.get("/main", response_class=HTMLResponse)
async def get_home(request: Request):
    # Session ID kontrolü
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = generate_session_id()
        request.session["session_id"] = session_id
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODELS,
        "model_descriptions": MODEL_DESCRIPTIONS,
        "available_models": AVAILABLE_MODELS,
        "saved_chats": load_chats(),
        "chat": None,
        "session_id": session_id
    })

# ----------------------- SORU SORMA -----------------------
@app.post("/", response_class=HTMLResponse) 
async def post_question(request: Request):
    # Session ID kontrolü
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = generate_session_id()
        request.session["session_id"] = session_id
    
    # Parse form data manually
    form = await request.form()
    question = form.get("question", "")
    models = form.getlist("models") 
    final_models = form.get("final_models", "")
    attachments = form.get("attachments")
    
    # Eğer final_models varsa (TÜM MODELLER + dinamik modeller), onu kullan
    if final_models and final_models.strip():
        try:
            import json
            models = json.loads(final_models)
            print(f"DEBUG: Final models (JSON): {models}")
        except Exception as e:
            print(f"DEBUG: JSON parse error: {str(e)}")
            # In case of error, use normal models
            pass
    
    # Conversation memory için ID oluştur
    conversation_id = form.get("conversation_id", "")
    if not conversation_id:
        conversation_id = generate_conversation_id(session_id)
    
    # Debug bilgileri dosyaya yaz
    debug_info = f"""
DEBUG LOG - {datetime.now()}
Form keys: {list(form.keys())}
Question: {question}
Models: {models}
Conversation ID: {conversation_id}
Attachments type: {type(attachments)}
Attachments: {attachments}
Has filename: {hasattr(attachments, 'filename') if attachments else False}
Filename: {getattr(attachments, 'filename', 'NO FILENAME') if attachments else 'NO ATTACHMENTS'}
---
"""
    
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(debug_info)
    
    print(f"DEBUG: Form keys: {list(form.keys())}")
    print(f"DEBUG: Question: {question}")
    print(f"DEBUG: Models: {models}")
    print(f"DEBUG: Conversation ID: {conversation_id}")
    print(f"DEBUG: Attachments type: {type(attachments)}")
    print(f"DEBUG: Attachments: {attachments}")
    
    if not question.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": MODELS,
            "model_descriptions": MODEL_DESCRIPTIONS,
            "available_models": AVAILABLE_MODELS,
            "saved_chats": load_chats(),
            "chat": None,
            "conversation_id": conversation_id
        })

    # Eski dosyaları ve memory'yi temizle
    cleanup_old_files()
    cleanup_old_memory()
    
    # Yüklenen dosyayı işle (tek dosya)
    file_contents = []
    attached_files = []
    
    print(f"DEBUG: File check started...")
    
    # UploadFile objesi mi kontrol et
    if attachments and hasattr(attachments, 'filename') and attachments.filename and attachments.filename.strip():
        print(f"DEBUG: File being checked: {attachments.filename}")
        try:
            # Dosyayı kaydet
            file_path = await save_uploaded_file(attachments)
            print(f"DEBUG: File saved: {file_path}")
            attached_files.append({"name": attachments.filename, "path": file_path})
            
            # Dosyadan metin çıkar
            content = extract_text_from_file(file_path)
            print(f"DEBUG: Extracted content length: {len(content)} characters")
            if content.strip():
                file_contents.append(f"📄 {attachments.filename}:\n{content}\n")
                print(f"DEBUG: File content added: {attachments.filename}")
        except Exception as e:
            print(f"DEBUG: File processing error: {str(e)}")
            file_contents.append(f"❌ {attachments.filename}: File processing error - {str(e)}\n")

    lang = detect_language(question)
    prompt_suffix = "Lütfen Türkçe yanıt verin." if lang == "tr" else "Please answer in English."

    # Conversation context (önceki mesajlar)
    conversation_context = get_conversation_context(conversation_id, max_messages=3)
    context_block = ""
    if conversation_context:
        context_block = f"\n\n📚 Previous Conversation History:\n{conversation_context}\n"
        print(f"DEBUG: Conversation context added, length: {len(conversation_context)}")

    time_sensitive = is_time_sensitive(question)
    search_snippets = await asyncio.to_thread(google_search_sync, question) if time_sensitive else ""
    search_block = f"\n\nSearch Snippets:\n{search_snippets}\n" if (time_sensitive and search_snippets) else ""
    
    # Dosya içeriklerini prompt'a ekle
    file_block = ""
    if file_contents:
        file_block = "\n\n�� Attached Files:\n" + "\n".join(file_contents)
        print(f"DEBUG: File block created, length: {len(file_block)}")
    else:
        print("DEBUG: No file content found")

    selected_models = MODELS if "__all__" in models else models

    # Dil bazında system prompt
    if lang == "tr":
        system_prompt = (
            "Sen hassas bir asistan. Kısa ve öz ol, varsayımları açıkça belirt ve halüsinasyonlardan kaçın. "
            "Emin değilsen söyle. Mümkün olduğunda yapılandırılmış madde işaretlerini tercih et. "
            "Bağlam farkında yanıtlar verirken konuşma geçmişini dikkate al."
        )
    else:
        system_prompt = (
            "You are a precise assistant. Be concise, cite assumptions explicitly, and avoid hallucinations. "
            "If unsure, say so. Prefer structured bullet points when helpful. "
            "Consider the conversation history when providing context-aware responses."
        )
    user_prompt = f"{question}\n{prompt_suffix}{context_block}{search_block}{file_block}"
    print(f"DEBUG: Final user prompt length: {len(user_prompt)}")
    print(f"DEBUG: User prompt content:\n{user_prompt}")

    # Paralel istekler (hangisi önce dönerse o üstte görünsün)
    tasks = [fetch_model_answer(m, system_prompt, user_prompt) for m in selected_models]
    responses = await asyncio.gather(*tasks)
    responses = list(responses)
    responses.sort(key=lambda x: x["elapsed"])

    # Combined Answer (GPT-5 ile)
    combined_inputs = "\n\n".join([f"### {r['model']}\n{r['text']}" for r in responses])

    # Dil bazında combined answer system prompt
    if lang == "tr":
        combine_system = (
            "Sen nihai sentezleyicisin. Birden fazla model yanıtını tek, tekrarsız ve "
            "doğru bir yanıtta birleştir. Çelişkileri çöz, kısa ama eksiksiz tut. Herhangi bir yanıt belirsizse "
            "belirsizliği kabul et. Gerçek doğruluğu koru."
        )
    else:
        combine_system = (
            "You are the final synthesizer. Merge multiple model answers into a single, non-redundant, "
            "accurate response. Resolve conflicts, keep it brief but complete. If any answer is uncertain, "
            "acknowledge uncertainty. Preserve factual correctness."
        )
    # Dil bazında combined answer user prompt
    if lang == "tr":
        combine_user = (
            f"{question}\n{prompt_suffix}{search_block}\n\n"
            "Aşağıda diğer modellerin yanıtları bulunmaktadır. "
            "Lütfen tekrarlamayın ve en iyi birleştirilmiş yanıtı verin:\n\n"
            f"{combined_inputs}"
        )
    else:
        combine_user = (
            f"{question}\n{prompt_suffix}{search_block}\n\n"
            "Below are the answers from other models. "
            "Please do not repeat and provide the best combined answer:\n\n"
            f"{combined_inputs}"
        )

    synthesis = "ERROR: Synthesis could not be obtained."
    synthesis_start_time = time.time()
    
    # Try GPT-5 first, then fallback to free models
    synthesis_models = [
        "openai/gpt-5-chat",  # Primary model
        "meta-llama/llama-3.3-70b-instruct:free",  # Fallback 1
        "mistralai/mistral-small-3.2-24b-instruct:free",  # Fallback 2
        "deepseek/deepseek-chat-v3-0324:free"  # Fallback 3
    ]
    
    synthesis = "ERROR: Synthesis could not be obtained from any model."
    
    for model in synthesis_models:
        try:
            print(f"DEBUG: Combined answer attempt with {model}")
            print(f"DEBUG: Prompt length sent to {model}: {len(combine_user)}")
            
            async with httpx.AsyncClient(timeout=150) as client:
                resp = await client.post(
                    OPENROUTER_API_URL,
                    headers=_headers(),
                    json={
                        "model": model,
                        "messages": _messages(combine_system, combine_user),
                        "stream": False,
                        "temperature": 0.2,
                    },
                )
                print(f"DEBUG: {model} response code: {resp.status_code}")
                data = resp.json()
                print(f"DEBUG: {model} response data: {data}")
                
                if resp.status_code == 200 and "error" not in data:
                    synthesis = _extract_text(data)
                    print(f"DEBUG: Successfully extracted synthesis from {model}: {synthesis[:100]}...")
                    break
                else:
                    print(f"DEBUG: {model} returned error: {data}")
                    continue
                    
        except Exception as e:
            print(f"DEBUG: Combined answer error with {model}: {str(e)}")
            continue
    
    # Calculate the duration of the combined answer
    synthesis_elapsed = round(time.time() - synthesis_start_time, 2)
    synthesis = f"[{synthesis_elapsed} sn] {synthesis}"

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    
    # ChatGPT style title - first 3-5 words
    def create_chat_title(question_text):
        import re
        # Clean up special characters and split into words
        clean_text = re.sub(r'[^\w\s]', '', question_text)
        words = clean_text.split()[:4]  # First 4 words
        title = ' '.join(words)
        if len(title) > 50:  # If too long, truncate
            title = title[:47] + "..."
        return title
    
    chat_title = create_chat_title(question)
    # Safe filename characters
    safe_filename = re.sub(r'[^\w\s-]', '', chat_title).strip()
    safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
    filename = f"{safe_filename}-{now.strftime('%m%d_%H%M')}.json"

    # Save to conversation memory
    ai_responses = [r["text"] for r in responses]
    add_to_memory(conversation_id, question, ai_responses, synthesis)
    
    # Add important information to global memory (name, preferences, etc.)
    if "step" in question.lower() or "my name" in question.lower() or "i" in question.lower():
        # Extract name
        name_match = re.search(r'(?:step|my name|i)\s+(?:what|is|who|zeynep|ahmet|mehmet|ayşe|fatma)', question.lower())
        if name_match:
            # Simple name extraction
            words = question.split()
            for i, word in enumerate(words):
                if word.lower() in ["step", "my name", "i"] and i + 1 < len(words):
                    name = words[i + 1]
                    if name.lower() not in ["what", "is", "who", "unremember", "remember"]:
                        add_to_global_memory("user_name", name)
                        print(f"DEBUG: Name added to global memory: {name}")
                        break
    
    chat = {
        "timestamp": timestamp,
        "question": question,
        "title": chat_title,
        "filename": filename,
        "lang": lang,
        "responses": responses,
        "synthesis": synthesis,
        "attached_files": [{"name": f["name"], "size": len(file_contents)} for f in attached_files] if attached_files else [],
        "conversation_id": conversation_id
    }
    save_chat_file(filename, chat)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODELS,
        "model_descriptions": MODEL_DESCRIPTIONS,
        "available_models": AVAILABLE_MODELS,
        "saved_chats": load_chats(),
        "chat": chat,
        "conversation_id": conversation_id
    })

# ----------------------- KAYITLI CHAT GETİR -----------------------
@app.get("/chat/{filename}", response_class=HTMLResponse)
async def get_chat(request: Request, filename: str):
    # Session ID kontrolü
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = generate_session_id()
        request.session["session_id"] = session_id
    
    with open(os.path.join(CHAT_DIR, filename), "r", encoding="utf-8") as f:
        chat = json.load(f)
    
    # Chat'te conversation_id yoksa oluştur
    if "conversation_id" not in chat:
        chat["conversation_id"] = generate_conversation_id(session_id)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODELS,
        "model_descriptions": MODEL_DESCRIPTIONS,
        "available_models": AVAILABLE_MODELS,
        "saved_chats": load_chats(),
        "chat": chat,
        "conversation_id": chat["conversation_id"]
    })

# ----------------------- CHAT SİL -----------------------
@app.delete("/delete/{filename}")
async def delete_chat(filename: str):
    try:
        delete_chat_file(filename)
        return {"status": "deleted"}
    except Exception as e:
        print(f"DEBUG: Chat deletion error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.delete("/delete_all")
async def delete_all():
    try:
        for f in os.listdir(CHAT_DIR):
            if f.endswith(".json"):
                delete_chat_file(f)
        return {"status": "all_deleted"}
    except Exception as e:
        print(f"DEBUG: Deleting all chats error: {str(e)}")
        return {"status": "error", "message": str(e)}

# ----------------------- CHAT EXPORT -----------------------
@app.get("/export/{filename}")
async def export_chat(filename: str):
    from fastapi.responses import PlainTextResponse
    
    try:
        with open(os.path.join(CHAT_DIR, filename), "r", encoding="utf-8") as f:
            chat = json.load(f)
        
        # Markdown formatında export
        export_content = f"""# {chat['question']}
**Date:** {chat['timestamp']}
**Language:** {'Turkish' if chat['lang'] == 'tr' else 'English'}

---

## Model Answers

"""
        
        for response in chat['responses']:
            export_content += f"""### {response['model']} ({response['elapsed']:.2f} seconds)
{response['text']}

---

"""
        
        export_content += f"""## Combined Answer
{chat['synthesis']}
"""
        
        return PlainTextResponse(
            export_content,
            headers={
                "Content-Disposition": f"attachment; filename={filename.replace('.json', '.md')}"
            }
        )
    except Exception as e:
        return {"error": str(e)}

# ----------------------- TEST ENDPOINT -----------------------
@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    print(f"TEST: File received - {file.filename}")
    print(f"TEST: Content type - {file.content_type}")
    content = await file.read()
    print(f"TEST: Content length - {len(content)} bytes")
    return {"filename": file.filename, "size": len(content), "content_preview": content[:100].decode('utf-8', errors='ignore')}

if __name__ == "__main__":
    import uvicorn
    print("\nAB AI MODEL: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
