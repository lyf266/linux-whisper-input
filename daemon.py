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
    "input_device": "",
    "silence_threshold": 0.01,
    "min_record_seconds": 0.5,
    "max_record_seconds": 120.0,
    "silence_duration": 3.0,
    "auto_stop_on_silence": True,
    "initial_prompt": "Hello, this is a sentence in English or Chinese. 你好，这是一个中英文混合的句子，包含正确的标点符号。",
    "proxy": "",
    "sound_enabled": True,
    "sound_volume": 0.15,
    "sound_on_start": True,
    "sound_on_stop": True,
    "sound_on_success": True,
    "sound_custom_start": "",
    "sound_custom_stop": "",
    "sound_custom_success": ""
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
import shutil
import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SOUNDS_DIR = os.path.join(DIR, "sounds")

def _safe_float(val, default: float) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_bool(val, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return default

def _find_audio_player() -> str | None:
    for p in ("paplay", "pw-play", "aplay"):
        path = shutil.which(p)
        if path:
            return path
    return None

AUDIO_PLAYER_BIN = _find_audio_player()

MODEL_SIZE = config.get("model_size", DEFAULT_CONFIG["model_size"])
DEVICE = config.get("device", DEFAULT_CONFIG["device"])
COMPUTE_TYPE = config.get("compute_type", DEFAULT_CONFIG["compute_type"])
LANGUAGE = config.get("language", DEFAULT_CONFIG["language"])
SAMPLE_RATE = int(config.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
CHANNELS = int(config.get("channels", DEFAULT_CONFIG["channels"]))
INPUT_DEVICE = str(config.get("input_device", DEFAULT_CONFIG["input_device"]))
SILENCE_THRESHOLD = _safe_float(config.get("silence_threshold"), DEFAULT_CONFIG["silence_threshold"])
MIN_RECORD_SECONDS = _safe_float(config.get("min_record_seconds"), DEFAULT_CONFIG["min_record_seconds"])
MAX_RECORD_SECONDS = _safe_float(config.get("max_record_seconds"), DEFAULT_CONFIG["max_record_seconds"])
SILENCE_DURATION = _safe_float(config.get("silence_duration"), DEFAULT_CONFIG["silence_duration"])
AUTO_STOP_ON_SILENCE = _safe_bool(config.get("auto_stop_on_silence"), DEFAULT_CONFIG["auto_stop_on_silence"])
INITIAL_PROMPT = config.get("initial_prompt", DEFAULT_CONFIG["initial_prompt"])
SOUND_ENABLED = _safe_bool(config.get("sound_enabled"), DEFAULT_CONFIG["sound_enabled"])
SOUND_VOLUME = _safe_float(config.get("sound_volume"), DEFAULT_CONFIG["sound_volume"])
SOUND_ON_START = _safe_bool(config.get("sound_on_start"), DEFAULT_CONFIG["sound_on_start"])
SOUND_ON_STOP = _safe_bool(config.get("sound_on_stop"), DEFAULT_CONFIG["sound_on_stop"])
SOUND_ON_SUCCESS = _safe_bool(config.get("sound_on_success"), DEFAULT_CONFIG["sound_on_success"])
SOUND_CUSTOM_START = str(config.get("sound_custom_start", DEFAULT_CONFIG["sound_custom_start"]))
SOUND_CUSTOM_STOP = str(config.get("sound_custom_stop", DEFAULT_CONFIG["sound_custom_stop"]))
SOUND_CUSTOM_SUCCESS = str(config.get("sound_custom_success", DEFAULT_CONFIG["sound_custom_success"]))

def reload_runtime_config():
    global config, LANGUAGE, SAMPLE_RATE, CHANNELS, INPUT_DEVICE, SILENCE_THRESHOLD
    global MIN_RECORD_SECONDS, MAX_RECORD_SECONDS, SILENCE_DURATION, AUTO_STOP_ON_SILENCE, INITIAL_PROMPT
    global SOUND_ENABLED, SOUND_VOLUME, SOUND_ON_START, SOUND_ON_STOP, SOUND_ON_SUCCESS
    global SOUND_CUSTOM_START, SOUND_CUSTOM_STOP, SOUND_CUSTOM_SUCCESS
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                new_cfg = json.load(f)
                config.update(new_cfg)
                LANGUAGE = config.get("language", DEFAULT_CONFIG["language"])
                SAMPLE_RATE = int(config.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
                CHANNELS = int(config.get("channels", DEFAULT_CONFIG["channels"]))
                INPUT_DEVICE = str(config.get("input_device", DEFAULT_CONFIG["input_device"]))
                SILENCE_THRESHOLD = _safe_float(config.get("silence_threshold"), DEFAULT_CONFIG["silence_threshold"])
                MIN_RECORD_SECONDS = _safe_float(config.get("min_record_seconds"), DEFAULT_CONFIG["min_record_seconds"])
                MAX_RECORD_SECONDS = _safe_float(config.get("max_record_seconds"), DEFAULT_CONFIG["max_record_seconds"])
                SILENCE_DURATION = _safe_float(config.get("silence_duration"), DEFAULT_CONFIG["silence_duration"])
                AUTO_STOP_ON_SILENCE = _safe_bool(config.get("auto_stop_on_silence"), DEFAULT_CONFIG["auto_stop_on_silence"])
                INITIAL_PROMPT = config.get("initial_prompt", DEFAULT_CONFIG["initial_prompt"])
                SOUND_ENABLED = _safe_bool(config.get("sound_enabled"), DEFAULT_CONFIG["sound_enabled"])
                SOUND_VOLUME = _safe_float(config.get("sound_volume"), DEFAULT_CONFIG["sound_volume"])
                SOUND_ON_START = _safe_bool(config.get("sound_on_start"), DEFAULT_CONFIG["sound_on_start"])
                SOUND_ON_STOP = _safe_bool(config.get("sound_on_stop"), DEFAULT_CONFIG["sound_on_stop"])
                SOUND_ON_SUCCESS = _safe_bool(config.get("sound_on_success"), DEFAULT_CONFIG["sound_on_success"])
                SOUND_CUSTOM_START = str(config.get("sound_custom_start", DEFAULT_CONFIG["sound_custom_start"]))
                SOUND_CUSTOM_STOP = str(config.get("sound_custom_stop", DEFAULT_CONFIG["sound_custom_stop"]))
                SOUND_CUSTOM_SUCCESS = str(config.get("sound_custom_success", DEFAULT_CONFIG["sound_custom_success"]))
                print(f"[{time.strftime('%H:%M:%S')}] Runtime config reloaded (lang={LANGUAGE}, dev={INPUT_DEVICE or 'default'}, silence_dur={SILENCE_DURATION}s, auto_stop={AUTO_STOP_ON_SILENCE}, sound={SOUND_ENABLED})")
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

# Audio prompt sound assets and playback
def _generate_wav_file(path: str, freqs: list[float] | tuple[float, ...], durations: list[float] | tuple[float, ...], sample_rate: int = 48000):
    chunks = []
    # 15ms lead-in silence to allow the audio server & DAC to uncork smoothly without popping
    chunks.append(np.zeros(int(sample_rate * 0.015), dtype=np.float32))
    
    for freq, dur in zip(freqs, durations):
        n_samples = int(sample_rate * dur)
        if n_samples <= 0:
            continue
        t = np.linspace(0, dur, n_samples, endpoint=False)
        attack_len = min(int(sample_rate * 0.012), n_samples // 4)
        decay_len = n_samples - attack_len
        attack = np.sin(np.pi * 0.5 * np.linspace(0, 1, attack_len)) ** 2
        decay = np.exp(-4.5 * np.linspace(0, 1, decay_len))
        env = np.concatenate([attack, decay])
        tone = 0.85 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * (freq * 2) * t)
        chunks.append(tone * env)
        
    # 60ms trailing silence to guarantee complete audio server buffer drain
    chunks.append(np.zeros(int(sample_rate * 0.060), dtype=np.float32))
    signal = np.concatenate(chunks)
    pcm16 = (signal * 32767).astype(np.int16)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

def ensure_sound_assets():
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    specs = {
        "start.wav": ([440.0, 587.0], [0.065, 0.075]),
        "stop.wav": ([587.0, 440.0], [0.065, 0.075]),
        "success.wav": ([659.0], [0.095]),
    }
    for filename, (freqs, durs) in specs.items():
        path = os.path.join(SOUNDS_DIR, filename)
        if not os.path.exists(path):
            try:
                _generate_wav_file(path, freqs, durs)
            except Exception as e:
                print(f"Error generating {filename}: {e}")

# Pre-generate standard sound assets on load
ensure_sound_assets()

def play_audio_file(file_path: str, volume: float = 0.15, timeout_sec: float = 3.0) -> bool:
    if not file_path or not os.path.isfile(file_path):
        return False
        
    vol = max(0.0, min(1.0, volume))
    if vol <= 0.0:
        return True

    # 1. Primary: Use system audio player (paplay / pw-play)
    if AUDIO_PLAYER_BIN:
        cmd = [AUDIO_PLAYER_BIN]
        basename = os.path.basename(AUDIO_PLAYER_BIN)
        if "paplay" in basename:
            cmd.extend([f"--volume={int(vol * 65536)}", file_path])
        elif "pw-play" in basename:
            cmd.extend([f"--volume={vol:.3f}", file_path])
        else:
            cmd.append(file_path)
            
        try:
            subprocess.run(
                cmd,
                check=False,
                timeout=timeout_sec,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.TimeoutExpired:
            print(f"Warning: Audio player timed out playing {file_path}")
            return False
        except Exception as e:
            print(f"Error playing sound file with {AUDIO_PLAYER_BIN}: {e}")

    # 2. Fallback: sounddevice playback if no system player available
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(data * vol, samplerate=sr)
            sd.wait()
            return True
    except Exception as e:
        print(f"Sounddevice fallback playback error: {e}")
        return False

def play_sound(event_type: str, block: bool = False):
    if not SOUND_ENABLED:
        return

    enabled_map = {
        "start": SOUND_ON_START,
        "stop": SOUND_ON_STOP,
        "success": SOUND_ON_SUCCESS,
    }
    if not enabled_map.get(event_type, False):
        return

    custom_map = {
        "start": SOUND_CUSTOM_START,
        "stop": SOUND_CUSTOM_STOP,
        "success": SOUND_CUSTOM_SUCCESS,
    }
    custom_path = custom_map.get(event_type, "")
    
    target_path = None
    if custom_path and os.path.isfile(custom_path):
        target_path = custom_path
    else:
        std_wav = os.path.join(SOUNDS_DIR, f"{event_type}.wav")
        if os.path.isfile(std_wav):
            target_path = std_wav

    if not target_path:
        return

    def _execute():
        play_audio_file(target_path, volume=SOUND_VOLUME, timeout_sec=3.0)

    if block:
        _execute()
    else:
        threading.Thread(target=_execute, daemon=True).start()

def rms(data: np.ndarray) -> float:
    if len(data) == 0:
        return 0.0
    return np.sqrt(np.mean(data ** 2))

class RecordingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_chunks = []
        self.device = None
        if INPUT_DEVICE and INPUT_DEVICE.strip():
            dev_str = INPUT_DEVICE.strip()
            try:
                dev_idx = int(dev_str) if dev_str.isdigit() else dev_str
                sd.query_devices(dev_idx, "input")
                self.device = dev_idx
                print(f"[{time.strftime('%H:%M:%S')}] Using configured input device: {self.device}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Configured input device '{INPUT_DEVICE}' not found ({e}). Using default device.")
                self.device = None

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
                device=self.device
            ):
                while self.is_recording:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Audio stream error: {e}")
            
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

def trim_trailing_silence(audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.02, pad_seconds: float = 0.3) -> np.ndarray:
    """
    Trim trailing silence from recorded audio to prevent Whisper hallucination loops.
    """
    if len(audio) == 0:
        return audio
        
    chunk_size = int(sample_rate * 0.1)  # 100ms blocks
    num_chunks = len(audio) // chunk_size
    if num_chunks == 0:
        return audio
        
    last_voice_idx = len(audio)
    for i in range(num_chunks - 1, -1, -1):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        chunk_rms = np.sqrt(np.mean(chunk ** 2))
        if chunk_rms > threshold:
            pad_samples = int(pad_seconds * sample_rate)
            last_voice_idx = min(len(audio), (i + 1) * chunk_size + pad_samples)
            break
            
    return audio[:last_voice_idx]

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
    audio_rms = rms(audio_data) if audio_data is not None else 0.0
    print(f"[{time.strftime('%H:%M:%S')}] Audio recording stopped. Duration: {duration:.2f}s, Level (RMS): {audio_rms:.5f}")
    
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
        
        # Trim trailing silence to prevent hallucination on tail padding
        orig_len = len(audio_float)
        trim_thresh = max(0.003, SILENCE_THRESHOLD / 2.0)
        audio_float = trim_trailing_silence(audio_float, sample_rate=SAMPLE_RATE, threshold=trim_thresh, pad_seconds=0.5)
        trimmed_len = len(audio_float)
        if orig_len > trimmed_len:
            print(f"[{time.strftime('%H:%M:%S')}] Trimmed trailing silence: {(orig_len - trimmed_len) / SAMPLE_RATE:.2f}s removed.")
        
        # Determine language parameter: None enables Whisper auto-detection (crucial for English & code)
        transcribe_lang = None if (not LANGUAGE or str(LANGUAGE).lower() in ("auto", "none")) else LANGUAGE
        
        # Run Whisper model with rigorous anti-hallucination settings
        segments, info = model.transcribe(
            audio_float,
            language=transcribe_lang,
            beam_size=5,
            condition_on_previous_text=False,  # CRITICAL: Disables runaway feedback loops from previous chunks
            repetition_penalty=1.15,            # Penalizes repetitive hallucinated token loops
            no_repeat_ngram_size=3,             # Forbids 3-gram repetitions (like "the gap, we call it the gap")
            compression_ratio_threshold=2.2,    # Filters out runaway repetitive segments
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.35,                 # Relaxed threshold to avoid cutting off quiet voices
                min_speech_duration_ms=200,
                min_silence_duration_ms=400,
                speech_pad_ms=250
            ),
            initial_prompt=INITIAL_PROMPT if INITIAL_PROMPT else None
        )
        
        print(f"[{time.strftime('%H:%M:%S')}] Language detected: '{info.language}' (prob: {info.language_probability:.2f})")
        
        # Combine segments preserving proper whitespace for English words and natural punctuation
        raw_text = "".join(seg.text for seg in segments).strip()
        result_text = raw_text
        
        if result_text:
            print(f"[{time.strftime('%H:%M:%S')}] Transcription finished successfully. Result: {result_text}")
            send_notification("✓ 识别完成", result_text, timeout_ms=2000)
            play_sound("success", block=False)
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
            if recorder:
                recorder.stop()
            play_sound("stop", block=False)
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
            
            play_sound("start", block=True)
            
            recorder = RecordingThread()
            recorder.start()
            
            silence_timer_thread = threading.Thread(target=silence_timer_loop, daemon=True)
            silence_timer_thread.start()
            
        elif state == "RECORDING":
            state = "TRANSCRIBING"
            print("Transitioning to TRANSCRIBING state (Manual Stop). Stopping audio stream...")
            if recorder:
                recorder.stop()
            play_sound("stop", block=False)
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
