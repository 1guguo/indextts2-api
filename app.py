# app.py (支持情感向量 / 自动情感 / 情感文本)
import os, uuid, subprocess, threading
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from indextts.infer_v2 import IndexTTS2

UPLOAD_DIR  = os.path.join(os.getcwd(), "uploads")
OUTPUT_DIR  = os.path.join(os.getcwd(), "outputs")
ALLOWED_EXT = {"wav", "mp3", "aac", "m4a", "ogg", "flac"}
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

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def to_wav_pcm_mono_22050(in_path: str, out_path: str):
    cmd = ["ffmpeg", "-y", "-i", in_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "22050", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

app = Flask(__name__)
app.secret_key = "indextts2-demo"

INDEX_HTML = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8"/>
  <title>IndexTTS2 语音克隆（扩展情感控制）</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:2rem;max-width:920px;margin:auto}
    .card{border:1px solid #e5e7eb;border-radius:14px;padding:1.25rem;margin-bottom:1rem; box-shadow:0 1px 6px rgba(0,0,0,.04)}
    label{display:block;margin:.5rem 0 .25rem 0;font-weight:600}
    input[type="text"], textarea, input[type="number"]{width:100%;padding:.65rem;border-radius:10px;border:1px solid #cbd5e1}
    input[type="file"]{margin:.25rem 0 1rem 0}
    button{background:#111827;color:#fff;border:none;border-radius:12px;padding:.7rem 1.1rem;cursor:pointer}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .hint{color:#6b7280;font-size:.9rem}
    .footer{color:#6b7280;margin-top:2rem;font-size:.85rem}
    audio{width:100%}
    .sliderwrap{display:flex;gap:12px;align-items:center}
    input[type="range"]{width:100%}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  </style>
</head>
<body>
  <h1>IndexTTS2 语音克隆（扩展情感控制）</h1>

  <div class="card">
    <form action="{{ url_for('synthesize') }}" method="post" enctype="multipart/form-data">
      <div class="row">
        <div>
          <label>音色参考音频（必选）</label>
          <input type="file" name="ref_audio" accept=".wav,.mp3,.aac,.m4a,.ogg,.flac" required>
          <div class="hint">用于克隆说话人音色，建议 3~10 秒、少噪声。</div>
        </div>
        <div>
          <label>情感参考音频（可选）</label>
          <input type="file" name="emo_audio" accept=".wav,.mp3,.aac,.m4a,.ogg,.flac">
          <div class="hint">用于控制语气/情绪，如“悲伤/激动”。</div>
        </div>
      </div>

      <label>合成文本</label>
      <textarea name="text" rows="3" required>哇塞！这个爆率也太高了！欧皇附体了！</textarea>

      <div class="grid2">
        <div>
          <label>情感权重（emo_alpha：0.0–1.0）</label>
          <div class="sliderwrap">
            <input type="range" min="0" max="1" step="0.05" value="1.0" oninput="emoVal.value=this.value" name="emo_alpha_range">
            <input type="number" min="0" max="1" step="0.01" value="1.0" name="emo_alpha_number" id="emoVal">
          </div>
          <div class="hint">值越大越贴近情感参考；0 等效于不使用情感参考。</div>
        </div>
        <div>
          <label>use_random（随机情感采样）</label>
          <select name="use_random">
            <option value="false" selected>false</option>
            <option value="true">true</option>
          </select>
          <div class="hint">开启会降低音色还原度。</div>
        </div>
      </div>

      <label>8 维情感向量（逗号分隔）</label>
      <input type="text" name="emo_vector" placeholder="格式：0,0,0,0,0,0,0.45,0">
      <div class="hint">顺序：[高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]；留空则不用。</div>

      <div class="grid2">
        <div>
          <label>use_emo_text（根据文本自动生情感）</label>
          <select name="use_emo_text">
            <option value="false" selected>false</option>
            <option value="true">true</option>
          </select>
        </div>
        <div>
          <label>情感文本（emo_text，可选）</label>
          <input type="text" name="emo_text" placeholder="例如：你吓死我了！你是鬼吗？">
          <div class="hint">启用 use_emo_text 时可提供情感文本，与朗读文本分离。</div>
        </div>
      </div>

      <div style="margin-top:1rem">
        <button type="submit">生成语音</button>
      </div>
    </form>
  </div>

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="card">
        {% for m in messages %}
          <div>{{ m|safe }}</div>
        {% endfor %}
      </div>
    {% endif %}
  {% endwith %}

  {% if audio_url %}
    <div class="card">
      <h3>生成结果</h3>
      <audio controls src="{{ audio_url }}"></audio>
      <p><a href="{{ audio_url }}" download>下载 WAV</a></p>
    </div>
  {% endif %}

  <div class="footer">提示：首次请求可能触发权重下载与模型加载，耗时较久；后续会快很多。</div>
  <script>
    const emoRange = document.querySelector('input[name="emo_alpha_range"]');
    const emoNum = document.querySelector('input[name="emo_alpha_number"]');
    emoRange.addEventListener('input', () => emoNum.value = emoRange.value);
    emoNum.addEventListener('input', () => {
      let v = parseFloat(emoNum.value);
      if (isNaN(v)) v = 1.0;
      v = Math.max(0, Math.min(1, v));
      emoNum.value = v.toFixed(2);
      emoRange.value = v.toFixed(2);
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, audio_url=None)

def _parse_emo_vector(s: str):
    """
    把 '0,0,0,0,0,0,0.45,0' -> [0.0,...]
    长度不足会右侧补 0；长度超出会截断到 8；非法值忽略。
    """
    if not s:
        return None
    try:
        vals = [float(x.strip()) for x in s.split(",")]
        vals = [v for v in vals if v == v]  # 去掉 NaN
        if not vals:
            return None
        if len(vals) < 8:
            vals += [0.0] * (8 - len(vals))
        return vals[:8]
    except Exception:
        return None

def _parse_bool(s: str):
    return str(s).lower() in ("1", "true", "yes", "y", "on")

def _parse_alpha(req):
    v = req.form.get("emo_alpha_number") or req.form.get("emo_alpha_range") or "1.0"
    try:
        v = float(v)
    except Exception:
        v = 1.0
    return max(0.0, min(1.0, v))

@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = (request.form.get("text") or "").strip()
    if not text:
        flash("请输入合成文本")
        return redirect(url_for("index"))

    emo_alpha   = _parse_alpha(request)
    use_random  = _parse_bool(request.form.get("use_random"))
    use_emo_txt = _parse_bool(request.form.get("use_emo_text"))
    emo_text    = (request.form.get("emo_text") or "").strip()
    emo_vector  = _parse_emo_vector(request.form.get("emo_vector", ""))

    # 音色参考
    if "ref_audio" not in request.files:
        flash("请上传音色参考音频")
        return redirect(url_for("index"))
    ref_f = request.files["ref_audio"]
    if not ref_f or not ref_f.filename or not allowed_file(ref_f.filename):
        flash("音色参考缺失或格式不支持")
        return redirect(url_for("index"))

    # 情感参考（可选）
    emo_f = request.files.get("emo_audio")

    uid = uuid.uuid4().hex
    ref_in  = os.path.join(UPLOAD_DIR, f"{uid}_ref_{secure_filename(ref_f.filename)}")
    ref_f.save(ref_in)
    ref_wav = os.path.join(UPLOAD_DIR, f"{uid}_ref.wav")
    to_wav_pcm_mono_22050(ref_in, ref_wav)

    emo_wav = None
    if emo_f and emo_f.filename:
        if not allowed_file(emo_f.filename):
            flash("情感参考格式不支持")
            return redirect(url_for("index"))
        emo_in  = os.path.join(UPLOAD_DIR, f"{uid}_emo_{secure_filename(emo_f.filename)}")
        emo_f.save(emo_in)
        emo_wav = os.path.join(UPLOAD_DIR, f"{uid}_emo.wav")
        to_wav_pcm_mono_22050(emo_in, emo_wav)

    out_path = os.path.join(OUTPUT_DIR, f"{uid}_gen.wav")

    # 构造 infer 的参数
    infer_kwargs = dict(
        spk_audio_prompt=ref_wav,
        text=text,
        output_path=out_path,
        verbose=False,
        emo_audio_prompt=emo_wav if emo_wav else None,
        emo_alpha=emo_alpha,
        use_random=use_random
    )
    # 仅当提供了向量时传递 emo_vector
    if emo_vector is not None:
        infer_kwargs["emo_vector"] = emo_vector
    # 仅当选择 use_emo_text 时传递 emo_text/use_emo_text
    if use_emo_txt:
        infer_kwargs["use_emo_text"] = True
        if emo_text:
            infer_kwargs["emo_text"] = emo_text

        # ✅ 打印输入参数，便于调试
    print("\n================= IndexTTS2 推理参数 =================")
    print(f"合成文本(text): {text}")
    print(f"音色参考(ref_audio): {ref_f.filename}")
    print(f"情感参考(emo_audio): {emo_f.filename if emo_f and emo_f.filename else None}")
    print(f"情感权重(emo_alpha): {emo_alpha}")
    print(f"随机情感(use_random): {use_random}")
    print(f"使用文本情感(use_emo_text): {use_emo_txt}")
    print(f"情感文本(emo_text): {emo_text}")
    print(f"情感向量(emo_vector): {emo_vector}")
    print("最终传入 tts.infer 的参数：")
    for k, v in infer_kwargs.items():
        print(f"  {k}: {v}")
    print("========================================================\n")

    try:
        with tts_lock:
            _ = tts.infer(**infer_kwargs)
    except Exception as e:
        flash(f"合成失败：{e}")
        return redirect(url_for("index"))

    audio_url = url_for("serve_output", filename=os.path.basename(out_path))
    msg = f"生成成功！emo_alpha={emo_alpha:.2f}"
    if emo_vector is not None:
        msg += f"，使用 emo_vector={emo_vector}"
    if use_random:
        msg += "（use_random=True）"
    if use_emo_txt:
        msg += "（use_emo_text=True"
        if "emo_text" in infer_kwargs:
            msg += f"，emo_text='{infer_kwargs['emo_text']}'"
        msg += "）"
    flash(msg)
    return render_template_string(INDEX_HTML, audio_url=audio_url)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)

