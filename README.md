# 🎙️ IndexTTS2 API 服务

本项目基于 [IndexTTS2](https://github.com/index-tts/index-tts) 官方模型，提供了 **FastAPI** 与 **Flask** 两种 Web 接口实现，支持：

- 🗣️ 语音克隆  
- 🎭 情感控制  
- 🔄 音色迁移  

> ⚠️ **重要提示**：`examples/` 目录中的示例音频文件 **不可直接使用**（部分未包含或格式不兼容），请务必上传您自己的参考音频。

---

## 📁 项目结构说明

### `app_fastapi.py`
✅ **推荐用于 API 调用**

基于 **FastAPI** 的高性能异步接口，适合：
- 前后端分离项目；
- 自动化批量调用；
- 与其他系统集成。

**启动命令：**
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```
**API 文档地址：**  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

### `app.py`
✅ **推荐用于本地调试或 Web 演示**

基于 **Flask** 的简易网页界面，支持表单上传和音频播放功能，适合快速测试。

**启动命令：**
```bash
python app.py
```
**访问地址：**  
👉 [http://localhost:7860](http://localhost:7860)

---

### `examples/` 目录
❌ **原始音频不可用！**

官方仓库中的示例音频未包含在本项目中，或因模型版本差异无法使用。  
请上传您自己的音频文件（建议 3～10 秒，背景噪音少）：

| 参数 | 说明 | 是否必填 |
|------|------|-----------|
| `ref_audio` | 音色参考（克隆说话人） | ✅ 必填 |
| `emo_audio` | 情感参考（控制语气） | ⚙️ 可选 |

---

## 🚀 快速开始

### 1️⃣ 准备模型文件
按官方指南下载模型到 `checkpoints/` 目录：

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

---

### 2️⃣ 启动服务（任选其一）

```bash
# 方式一：FastAPI（API 接口）
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000

# 方式二：Flask（Web 界面）
python app.py
```

---

## ⚙️ 接口参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `text` | str | 要合成的文本（必填） |
| `ref_audio` | file | 音色参考音频（必填） |
| `emo_audio` | file | 情感参考音频（可选） |
| `emo_vector` | list[float] | 8 维情感向量 `[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]` |
| `emo_alpha` | float | 情感权重（0.0–1.0） |
| `use_random` | bool | 随机情感采样 |
| `use_emo_text` | bool | 是否根据文本自动生情感 |
| `emo_text` | str | 单独的情感文本 |

---

## ⚠️ 注意事项

- 首次运行较慢（模型加载 + 权重初始化）；
- 支持的音频格式：`.wav`, `.mp3`, `.flac`, `.aac`, `.m4a`, `.ogg`；
- 请勿上传空文件或非音频文件；
- 情感控制效果取决于参考音频的情绪表现；
- 所有生成的音频文件默认保存在 `outputs/` 目录。

---

## 🧠 技术要点

- Flask 版本提供可视化界面与文件上传；
- FastAPI 版本提供 RESTful JSON 接口；
- 支持情感向量（`emo_vector`）与文本情感 (`use_emo_text`)；
- 内置 FFmpeg 转码逻辑（自动转换为单声道 22.05kHz PCM WAV）；
- 错误处理更健壮：空文件、非法格式、无效情感音频将被忽略并提示。

---

## 🤝 致谢

本项目基于 **IndexTTS2** 开发，感谢 **Bilibili IndexSpeech 团队** 的开源贡献！  
如需商业合作，请联系官方邮箱：  
📧 **indexspeech@bilibili.com**
