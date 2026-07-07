import os
import sys
import shutil

# Ensure all proxy environment variables are cleared for direct connection to ModelScope
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    if var in os.environ:
        del os.environ[var]

# Import modelscope after proxy environment variables are cleared
try:
    from modelscope import snapshot_download
except ImportError:
    print("❌ Error: modelscope package is not installed. Please run: ./venv/bin/pip install modelscope")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "model_scope_cache")
TARGET_DIR = os.path.join(PROJECT_DIR, "model")

print("Downloading model mobiuslabsgmbh/faster-whisper-large-v3-turbo from ModelScope...")
try:
    # 1. Download to temporary cache directory
    model_dir = snapshot_download(
        model_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        cache_dir=CACHE_DIR
    )
    print(f"\nModel downloaded to temporary cache: {model_dir}")
    
    # 2. Move to final target directory
    print(f"Moving model files to: {TARGET_DIR}...")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        
    shutil.move(model_dir, TARGET_DIR)
    
    # 3. Clean up temporary cache
    print("Cleaning up temporary cache...")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        
    print(f"\n🎉 Success! Model is fully cached locally at: {TARGET_DIR}")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error downloading from ModelScope: {e}")
    sys.exit(1)
