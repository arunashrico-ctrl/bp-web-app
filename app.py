from flask import Flask, render_template, request, jsonify
import os
import json
from bp_model import predict_bp_dual_roi

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():

    video = request.files['video']
    cheek_roi = json.loads(request.form['cheek_roi'])
    palm_roi = json.loads(request.form['palm_roi'])

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    result = predict_bp_dual_roi(video_path, cheek_roi, palm_roi)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)