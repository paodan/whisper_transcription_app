from flask import Flask, request, jsonify, render_template
import whisper
import os
import threading
import time
import ssl
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
start_time = None
transcribing = False
# 这里使用了 Whisper 模型，支持多种语言
# 你可以根据需要选择其他模型
# 例如：model = whisper.load_model("tiny")（不支持长文本）
# 但速度更快
# 你可以在 https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads 上找到更多模型
model = whisper.load_model("base")  # 可选 tiny/base/small/medium/large
UPLOAD_FOLDER = os.path.join("static", "uploaded")
print(f"Upload folder: {UPLOAD_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress = {"percent": 0}

# ✅ 加载 LED summarizer（支持长文本）
modelname = "allenai/led-base-16384"
# 这里使用了 LED 模型，支持长文本摘要
# 你可以根据需要选择其他模型
# 例如：modelname = "facebook/bart-large-cnn"（不支持长文本）
# 但速度更快
# 你可以在 https://huggingface.co/models?pipeline_tag=summarization&sort=downloads 上找到更多模型
# 这里使用了 LED 模型，支持长文本摘要
tokenizer = LEDTokenizer.from_pretrained(modelname)
summarizer_model = LEDForConditionalGeneration.from_pretrained(modelname)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer_model.to(device)
summarizer_model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/progress")
def get_progress():
    return jsonify(progress)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400
    audio = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(file_path)

    def process():
        start_time = time.time()
        progress["percent"] = 10
        result = model.transcribe(file_path)
        progress["percent"] = 60
        transcript = result["text"]

        summary = generate_summary(transcript)
        progress["percent"] = 90

        save_path = os.path.join(UPLOAD_FOLDER, f"{audio.filename}.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("--- Transcript ---\n")
            f.write(transcript + "\n\n")
            f.write("--- Summary ---\n")
            f.write(summary)

        progress["text"] = transcript
        progress["summary"] = summary
        progress["percent"] = 100
        progress["time"] = round(time.time() - start_time, 2)
        progress["audio_url"] = f"/static/uploaded/{audio.filename}"

        #os.remove(file_path)

    threading.Thread(target=process).start()

    return jsonify({"status": "processing"})

def generate_summary(transcript):
    try:
        inputs = tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=16384,
            padding="max_length"
        )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        global_attention_mask = global_attention_mask.to(device)

        summary_ids = summarizer_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_length=256,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
