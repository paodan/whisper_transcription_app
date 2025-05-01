
from flask import Flask, request, jsonify, render_template
import whisper
import os
import openai
import threading
import time

app = Flask(__name__)
model = whisper.load_model("base")  # 可选 tiny/base/small/medium/large
#UPLOAD_FOLDER = "temp"
UPLOAD_FOLDER = os.path.join("static", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

progress = {"percent": 0}  # 全局进度变量

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
        progress["audio_url"] = f"/static/uploaded/{audio.filename}"

        #os.remove(file_path)

    threading.Thread(target=process).start()

    return jsonify({"status": "processing"})

def generate_summary(transcript):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize meeting transcripts in a concise way."},
                {"role": "user", "content": transcript}
            ],
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)




# from flask import Flask, request, jsonify, render_template
# import whisper
# import os
# import openai
# from openai import OpenAI

# app = Flask(__name__)
# model = whisper.load_model("base")  # 可选 tiny/base/small/medium/large
# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # 设置你的 OpenAI API Key（可从环境变量读取）
# # openai.api_key = os.getenv("OPENAI_API_KEY")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/transcribe", methods=["POST"])
# # def transcribe():
# #     if "audio" not in request.files:
# #         return jsonify({"error": "No audio uploaded"}), 400
# #     audio = request.files["audio"]
# #     file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
# #     audio.save(file_path)

# #     result = model.transcribe(file_path)
# #     os.remove(file_path)

# #     transcript = result["text"]
# #     summary = generate_summary(transcript)

# #     return jsonify({"text": transcript, "summary": summary})
# def transcribe():
#     if "audio" not in request.files:
#         return jsonify({"error": "No audio uploaded"}), 400
#     audio = request.files["audio"]
#     file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
#     audio.save(file_path)

#     result = model.transcribe(file_path)
#     os.remove(file_path)

#     transcript = result["text"]
#     summary = generate_summary(transcript)

#     # 保存到 txt 文件
#     save_path = os.path.join(UPLOAD_FOLDER, f"{audio.filename}.txt")
#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write("--- Transcript ---\n")
#         f.write(transcript + "\n\n")
#         f.write("--- Summary ---\n")
#         f.write(summary)

#     return jsonify({"text": transcript, "summary": summary})


# client = OpenAI()  # 使用环境变量中的 OPENAI_API_KEY 自动认证

# def generate_summary(transcript):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You summarize meeting transcripts in a concise way."},
#                 {"role": "user", "content": transcript}
#             ],
#             max_tokens=500
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error generating summary: {str(e)}"
    

# if __name__ == "__main__":
#     app.run(debug=True)
