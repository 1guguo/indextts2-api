# app_fastapi.py
import os
import uuid
import subprocess
import threading
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename  # 用于安全文件名
from indextts.infer_v2 import IndexTTS2

# ----------------- 基础配置 -----------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ALLOWED_EXTENSIONS = {"wav", "mp3", "aac", "m4a", "ogg", "flac"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tts_lock = threading.Lock()
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False
)

app = FastAPI(title="IndexTTS2 FastAPI 语音克隆接口", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- 工具函数 -----------------
def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否在允许列表中"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def to_wav_pcm_mono_22050(in_path: str, out_path: str):
    """用 ffmpeg 转成标准 wav 格式"""
    cmd = ["ffmpeg", "-y", "-i", in_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "22050", out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "未知错误"
        raise RuntimeError(f"FFmpeg 转换失败: {stderr}") from e

def _parse_emo_vector(s: str):
    if not s:
        return None
    try:
        vals = [float(x.strip()) for x in s.split(",")]
        vals = [v for v in vals if v == v]  # 过滤 NaN
        if len(vals) < 8:
            vals += [0.0] * (8 - len(vals))
        return vals[:8]
    except Exception:
        return None

def _parse_bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")

# ----------------- API 路由 -----------------

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "message": "IndexTTS2 service is running"})

@app.post("/synthesize")
async def synthesize(
    text: str = Form(..., description="要合成的文本"),
    emo_alpha: float = Form(1.0, description="情感权重 0~1"),
    use_random: bool = Form(False, description="是否使用随机情感采样"),
    use_emo_text: bool = Form(False, description="是否根据文本生成情感向量"),
    emo_text: str = Form("", description="情感文本（可选）"),
    emo_vector: str = Form("", description="8维情感向量，逗号分隔（可选）"),
    ref_audio: UploadFile = File(..., description="音色参考音频"),
    emo_audio: UploadFile = File(None, description="情感参考音频（可选）")
):
    try:
        # 校验文本
        text = text.strip()
        if not text:
            raise ValueError("合成文本不能为空")

        # 校验并保存音色参考音频（必填）
        if not ref_audio.filename:
            raise ValueError("音色参考音频文件名为空")
        if not allowed_file(ref_audio.filename):
            raise ValueError("音色参考音频格式不支持，请上传 WAV/MP3/FLAC/AAC/M4A/OGG 文件")
        
        ref_content = await ref_audio.read()
        if not ref_content:
            raise ValueError("音色参考音频为空，请上传有效的音频文件")

        uid = uuid.uuid4().hex
        safe_ref_name = secure_filename(ref_audio.filename)
        ref_in = os.path.join(UPLOAD_DIR, f"{uid}_ref_{safe_ref_name}")
        with open(ref_in, "wb") as f:
            f.write(ref_content)
        ref_wav = os.path.join(UPLOAD_DIR, f"{uid}_ref.wav")
        to_wav_pcm_mono_22050(ref_in, ref_wav)

        # 处理情感参考音频（可选，仅当有效时处理）
        emo_wav = None
        if emo_audio and emo_audio.filename:
            if not allowed_file(emo_audio.filename):
                raise ValueError("情感参考音频格式不支持")
            emo_content = await emo_audio.read()
            if not emo_content:
                raise ValueError("情感参考音频为空")
            safe_emo_name = secure_filename(emo_audio.filename)
            emo_in = os.path.join(UPLOAD_DIR, f"{uid}_emo_{safe_emo_name}")
            with open(emo_in, "wb") as f:
                f.write(emo_content)
            emo_wav = os.path.join(UPLOAD_DIR, f"{uid}_emo.wav")
            to_wav_pcm_mono_22050(emo_in, emo_wav)

        # 构造输出路径
        out_path = os.path.join(OUTPUT_DIR, f"{uid}_gen.wav")

        # 构造推理参数
        emo_vec = _parse_emo_vector(emo_vector)
        kwargs = dict(
            spk_audio_prompt=ref_wav,
            text=text,
            output_path=out_path,
            verbose=False,
            emo_audio_prompt=emo_wav,
            emo_alpha=float(emo_alpha),
            use_random=_parse_bool(use_random),
        )
        if emo_vec is not None:
            kwargs["emo_vector"] = emo_vec
        if use_emo_text:
            kwargs["use_emo_text"] = True
            if emo_text.strip():
                kwargs["emo_text"] = emo_text.strip()

        # 推理
        with tts_lock:
            _ = tts.infer(**kwargs)

        return FileResponse(out_path, media_type="audio/wav", filename="gen.wav")

    except ValueError as e:
        # 用户输入错误（400）
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        # 服务器内部错误（500）
        return JSONResponse({"error": str(e)}, status_code=500)


# 启动方式 uvicorn app_fastapi:app --host 0.0.0.0 --port 8000