from pathlib import Path
import shutil
import subprocess

print("🚀 Initializing ORBIT Assistant...")

# -----------------------------
# Directories
# -----------------------------
folders = [
    "backend/app",
    "backend/app/api",
    "backend/app/services",
    "backend/app/llm",
    "backend/app/vectorstore",
    "backend/app/ingest",
    "backend/app/config",
    "backend/app/models",
    "backend/docs/apex",
    "backend/docs/userstories",
    "backend/docs/pdf",
    "backend/chroma_db"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# -----------------------------
# requirements.txt
# -----------------------------
requirements = """fastapi
uvicorn
langchain
langchain-community
langchain-ollama
chromadb
pypdf
python-docx
python-dotenv
"""

req_file = Path("backend/requirements.txt")

if not req_file.exists():
    req_file.write_text(requirements)
    print("✅ requirements.txt created")

# -----------------------------
# .env
# -----------------------------
env_content = """LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
EMBEDDING_MODEL=nomic-embed-text
VECTOR_STORE=chroma
CHROMA_PATH=./chroma_db
"""

env_file = Path("backend/.env")

if not env_file.exists():
    env_file.write_text(env_content)
    print("✅ .env created")

# -----------------------------
# main.py
# -----------------------------
main_content = '''from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def health():
    return {
        "status": "OK"
    }
'''

main_file = Path("backend/main.py")

if not main_file.exists():
    main_file.write_text(main_content)
    print("✅ main.py created")

# -----------------------------
# __init__.py files
# -----------------------------
for package in [
    "backend/app",
    "backend/app/api",
    "backend/app/services",
    "backend/app/llm",
    "backend/app/vectorstore",
    "backend/app/ingest",
    "backend/app/config",
    "backend/app/models"
]:
    init_file = Path(package) / "__init__.py"
    init_file.touch(exist_ok=True)

# -----------------------------
# Check Ollama
# -----------------------------
if shutil.which("ollama"):
    print("✅ Ollama detected")

    try:
        models = subprocess.check_output(
            ["ollama", "list"],
            text=True
        )

        if "qwen3:8b" not in models:
            print("⬇️ Pulling qwen3:8b ...")
            subprocess.run(["ollama", "pull", "qwen3:8b"])

        if "nomic-embed-text" not in models:
            print("⬇️ Pulling nomic-embed-text ...")
            subprocess.run(["ollama", "pull", "nomic-embed-text"])

    except Exception as e:
        print(f"⚠️ Could not verify models: {e}")

else:
    print("⚠️ Ollama not found.")
    print("Install from https://ollama.com")

print()
print("🎉 ORBIT Assistant initialized")
print()
print("Next steps:")
print("1. cd backend")
print("2. pip install -r requirements.txt")
print("3. uvicorn main:app --reload")
print()
print("Swagger UI:")
print("http://localhost:8000/docs")