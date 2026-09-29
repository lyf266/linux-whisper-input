#!/usr/bin/env python3
import os
import sys
import json

# Load config and set proxy immediately at startup
DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIR, "config.json")

DEFAULT_CONFIG = {
    "model_size": "/home/f1are/voice-input-service/model",
    "device": "cuda",
    "compute_type": "float16",
    "language": "auto",
    "sample_rate": 16000,
    "channels": 1,
    "silence_threshold": 0.04,
    "min_record_seconds": 0.5,
    "max_record_seconds": 120.0,
    "silence_duration": 3.0,
    "auto_stop_on_silence": True,
    "initial_prompt": "Hello, this is a sentence in English or Chinese. 你好，这是一个中英文混合的句子，包含正确的标点符号。",
    "proxy": ""
}

config = DEFAULT_CONFIG.copy()
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    except Exception as e:
        print(f"Error loading config.json, using defaults: {e}")

PROXY = config.get("proxy")
if PROXY:
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY
    # If using local loopback proxy, also map all_proxy/ALL_PROXY to socks5 protocol
    os.environ["ALL_PROXY"] = PROXY.replace("http://", "socks5://") if "127.0.0.1" in PROXY else PROXY
    os.environ["all_proxy"] = os.environ["ALL_PROXY"]
    # Set Hugging Face mirror endpoint for fast/stable downloads
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"Set proxy: {PROXY}")
    print("Set HF_ENDPOINT: https://hf-mirror.com")

import socket
import threading
import time
import subprocess
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = config.get("model_size", DEFAULT_CONFIG["model_size"])
DEVICE = config.get("device", DEFAULT_CONFIG["device"])
COMPUTE_TYPE = config.get("compute_type", DEFAULT_CONFIG["compute_type"])
LANGUAGE = config.get("language", DEFAULT_CONFIG["language"])
SAMPLE_RATE = config.get("sample_rate", DEFAULT_CONFIG["sample_rate"])
CHANNELS = config.get("channels", DEFAULT_CONFIG["channels"])
SILENCE_THRESHOLD = config.get("silence_threshold", DEFAULT_CONFIG["silence_threshold"])
MIN_RECORD_SECONDS = config.get("min_record_seconds", DEFAULT_CONFIG["min_record_seconds"])
MAX_RECORD_SECONDS = config.get("max_record_seconds", DEFAULT_CONFIG["max_record_seconds"])
SILENCE_DURATION = config.get("silence_duration", DEFAULT_CONFIG["silence_duration"])
AUTO_STOP_ON_SILENCE = config.get("auto_stop_on_silence", DEFAULT_CONFIG["auto_stop_on_silence"])
INITIAL_PROMPT = config.get("initial_prompt", DEFAULT_CONFIG["initial_prompt"])

def reload_runtime_config():
    global config, LANGUAGE, SAMPLE_RATE, CHANNELS, SILENCE_THRESHOLD
    global MIN_RECORD_SECONDS, MAX_RECORD_SECONDS, SILENCE_DURATION, AUTO_STOP_ON_SILENCE, INITIAL_PROMPT
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                new_cfg = json.load(f)
                config.update(new_cfg)
                LANGUAGE = config.get("language", DEFAULT_CONFIG["language"])
                SAMPLE_RATE = config.get("sample_rate", DEFAULT_CONFIG["sample_rate"])
                CHANNELS = config.get("channels", DEFAULT_CONFIG["channels"])
                SILENCE_THRESHOLD = float(config.get("silence_threshold", DEFAULT_CONFIG["silence_threshold"]))
                MIN_RECORD_SECONDS = float(config.get("min_record_seconds", DEFAULT_CONFIG["min_record_seconds"]))
                MAX_RECORD_SECONDS = float(config.get("max_record_seconds", DEFAULT_CONFIG["max_record_seconds"]))
                SILENCE_DURATION = float(config.get("silence_duration", DEFAULT_CONFIG["silence_duration"]))
                AUTO_STOP_ON_SILENCE = bool(config.get("auto_stop_on_silence", DEFAULT_CONFIG["auto_stop_on_silence"]))
                INITIAL_PROMPT = config.get("initial_prompt", DEFAULT_CONFIG["initial_prompt"])
                print(f"[{time.strftime('%H:%M:%S')}] Runtime config reloaded (lang={LANGUAGE}, silence_dur={SILENCE_DURATION}s, auto_stop={AUTO_STOP_ON_SILENCE})")
        except Exception as e:
            print(f"Error reloading config: {e}")

# State variables
state_lock = threading.Lock()
state = "IDLE"  # IDLE, RECORDING, TRANSCRIBING

recorder = None
silence_timer_thread = None
model = None

# Notification helper
def send_notification(title, message="", timeout_ms=1000):
    try:
        # Use a synchronous group key to replace existing notifications
        subprocess.run([
            "notify-send",
            "-t", str(timeout_ms),
            "-h", "string:x-canonical-private-synchronous:voice-input",
            title,
            message
        ], check=False)
    except Exception as e:
        print(f"Notification error: {e}")

def rms(data: np.ndarray) -> float:
    if len(data) == 0:
        return 0.0
    return np.sqrt(np.mean(data ** 2))

class RecordingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_chunks = []
        self.device_name = "sysdefault"
        try:
            sd.query_devices(self.device_name, "input")
        except Exception:
            self.device_name = None

    def run(self):
        self.is_recording = True
        self.audio_chunks = []
        
        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_chunks.append(indata.copy())
                
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
                callback=callback,
                device=self.device_name
            ):
                while self.is_recording:
                    time.sleep(0.05)
        except Exception as e:
            print(f"Audio stream error: {e}")
            
    def stop(self):
        self.is_recording = False

def type_text(text: str):
    wayland = os.environ.get("WAYLAND_DISPLAY")
    
    # 1. Copy to clipboard
    if wayland:
        try:
            subprocess.run(["wl-copy", text], check=True, timeout=5)
        except Exception as e:
            print(f"wl-copy failed: {e}")
    else:
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        except Exception as e:
            print(f"xclip failed: {e}")
            
    # Short sleep to make sure clip is registered
    time.sleep(0.12)
    
    # 2. Trigger paste
    if wayland:
        # Try ydotool
        try:
            subprocess.run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], check=True, timeout=3)
            print("Pasted via ydotool")
            return
        except Exception as e:
            print(f"ydotool failed: {e}")
            
        # Try wtype
        try:
            subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], check=True, timeout=3)
            print("Pasted via wtype")
            return
        except Exception as e:
            print(f"wtype failed: {e}")
    else:
        # X11 fallback
        try:
            subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=True, timeout=3)
            print("Pasted via xdotool")
            return
        except Exception as e:
            print(f"xdotool failed: {e}")

# Transcribe function
def stop_and_transcribe(reason="auto"):
    global state, recorder
    print(f"[{time.strftime('%H:%M:%S')}] stop_and_transcribe starting (reason: {reason})...")
    
    # Stop recording
    audio_data = None
    if recorder:
        print("Stopping audio recording thread...")
        recorder.stop()
        recorder.join(timeout=3)
        if recorder.audio_chunks:
            audio_data = np.concatenate(recorder.audio_chunks, axis=0)
        
    duration = len(audio_data) / SAMPLE_RATE if audio_data is not None else 0
    print(f"[{time.strftime('%H:%M:%S')}] Audio recording stopped. Duration: {duration:.2f}s")
    
    if duration < MIN_RECORD_SECONDS or audio_data is None:
        print(f"[{time.strftime('%H:%M:%S')}] Skipping: duration ({duration:.2f}s) is shorter than minimum ({MIN_RECORD_SECONDS}s)")
        send_notification("⚠️ 录音时间太短", "请说长一点。")
        with state_lock:
            state = "IDLE"
        return
        
    send_notification("🔄 正在识别...", "请稍候...")
    print(f"[{time.strftime('%H:%M:%S')}] Starting Whisper model inference...")
    
    try:
        audio_float = audio_data.flatten()
        
        # Determine language parameter: None enables Whisper auto-detection (crucial for English & code)
        transcribe_lang = None if (not LANGUAGE or str(LANGUAGE).lower() in ("auto", "none")) else LANGUAGE
        
        # Run Whisper model
        segments, info = model.transcribe(
            audio_float,
            language=transcribe_lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(speech_pad_ms=250),
            initial_prompt=INITIAL_PROMPT if INITIAL_PROMPT else None
        )
        
        print(f"[{time.strftime('%H:%M:%S')}] Language detected: '{info.language}' (prob: {info.language_probability:.2f})")
        
        # Combine segments preserving proper whitespace for English words and natural punctuation
        raw_text = "".join(seg.text for seg in segments).strip()
        result_text = raw_text
        
        if result_text:
            print(f"[{time.strftime('%H:%M:%S')}] Transcription finished successfully. Result: {result_text}")
            send_notification("✓ 识别完成", result_text, timeout_ms=2000)
            type_text(result_text)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Transcription finished. No speech/text detected.")
            send_notification("⚠️ 未识别到文字", "请重新输入")
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Transcription error: {e}")
        send_notification("❌ 识别出错", str(e))
        
    with state_lock:
        state = "IDLE"
    print(f"[{time.strftime('%H:%M:%S')}] State reset to IDLE.")

def silence_timer_loop():
    global state
    silence_start = None
    recording_start = time.time()
    
    while True:
        time.sleep(0.1)
        with state_lock:
            if state != "RECORDING":
                break
        
        elapsed = time.time() - recording_start
        
        # Max duration timeout (hard cap)
        if elapsed >= MAX_RECORD_SECONDS:
            print(f"Max recording duration ({MAX_RECORD_SECONDS}s) reached")
            trigger_stop(reason="max_duration")
            break
            
        # If auto_stop_on_silence is False, stay in manual mode
        if not AUTO_STOP_ON_SILENCE:
            continue
            
        # VAD silence check
        if recorder and len(recorder.audio_chunks) >= 3:
            recent = np.concatenate(recorder.audio_chunks[-3:])
            level = rms(recent)
            
            if level > SILENCE_THRESHOLD:
                silence_start = None
            else:
                if silence_start is None:
                    if elapsed > MIN_RECORD_SECONDS:
                        silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    print(f"Silence detected ({SILENCE_DURATION:.1f}s), auto-stopping...")
                    trigger_stop(reason="silence")
                    break

def trigger_stop(reason="auto"):
    global state
    with state_lock:
        if state == "RECORDING":
            state = "TRANSCRIBING"
            threading.Thread(target=stop_and_transcribe, kwargs={"reason": reason}, daemon=True).start()

def handle_toggle():
    global state, recorder, silence_timer_thread
    print(f"[{time.strftime('%H:%M:%S')}] Toggle received. Current state: {state}")
    with state_lock:
        if state == "IDLE":
            reload_runtime_config()
            state = "RECORDING"
            print("Transitioning to RECORDING state. Starting audio stream...")
            
            hint = "再次按快捷键手动发送" + (f"，或停顿{SILENCE_DURATION:g}秒自动发送" if AUTO_STOP_ON_SILENCE else "")
            send_notification("🎤 正在录音...", hint, timeout_ms=3000)
            
            recorder = RecordingThread()
            recorder.start()
            
            silence_timer_thread = threading.Thread(target=silence_timer_loop, daemon=True)
            silence_timer_thread.start()
            
        elif state == "RECORDING":
            state = "TRANSCRIBING"
            print("Transitioning to TRANSCRIBING state (Manual Stop). Stopping audio stream...")
            send_notification("⏹️ 结束录音", "正在识别，请稍候...", timeout_ms=1500)
            threading.Thread(target=stop_and_transcribe, kwargs={"reason": "manual"}, daemon=True).start()
            
        elif state == "TRANSCRIBING":
            print("Currently transcribing, toggle ignored.")

def main():
    global model
    
    print("Starting Voice Input Daemon...")
    print(f"Model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
    
    # Load model
    try:
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("Whisper Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try CPU fallback
        if DEVICE == "cuda":
            print("Attempting CPU fallback...")
            try:
                model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                print("Model loaded on CPU fallback.")
            except Exception as e2:
                print(f"CPU fallback failed: {e2}")
                sys.exit(1)
        else:
            sys.exit(1)
            
    # Setup UNIX socket
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    socket_path = os.path.join(runtime_dir, "voice-input.sock")
    
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass
            
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    
    # Restrict socket permissions to owner only
    os.chmod(socket_path, 0o600)
    
    print(f"Daemon listening on socket: {socket_path}")
    send_notification("🎙️ 语音输入服务已就绪", "快捷键触发即可开始录音")
    
    try:
        while True:
            conn, _ = server.accept()
            try:
                raw_data = conn.recv(4096).decode("utf-8").strip()
                if not raw_data:
                    continue
                try:
                    payload = json.loads(raw_data)
                    command = payload.get("command")
                    env = payload.get("env", {})
                    # Update environment from active client
                    for k, v in env.items():
                        if v:
                            os.environ[k] = v
                except json.JSONDecodeError:
                    command = raw_data
                
                print(f"[{time.strftime('%H:%M:%S')}] Socket received: {command}")
                if command == "toggle":
                    handle_toggle()
            except Exception as e:
                print(f"Error handling socket client: {e}")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("Shutting down daemon...")
    finally:
        if os.path.exists(socket_path):
            os.unlink(socket_path)

if __name__ == "__main__":
    main()
