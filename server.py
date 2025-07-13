from flask import Flask, request
from deepface import DeepFace
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return json.dumps({'error': 'No image uploaded'}, ensure_ascii=False), 400

    image = request.files['image']
    if image.filename == "":
        return json.dumps({'error': 'Empty filename'}, ensure_ascii=False), 400

    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    try:
        mtcnn_result = DeepFace.analyze(
            img_path=filepath,
            actions=['emotion'],
            enforce_detection=False,
            align=True,
            detector_backend='mtcnn',  # 使用更精准的人脸检测器
        )
        retinaface_result = DeepFace.analyze(
            img_path=filepath,
            actions=['emotion'],
            enforce_detection=False,
            align=True,
            detector_backend='retinaface',  # 使用更精准的人脸检测器
        )
        media_pipe_result = DeepFace.analyze(
            img_path=filepath,
            actions=['emotion'],
            enforce_detection=False,
            align=True,
            detector_backend='mediapipe',  # 使用更精准的人脸检测器
        )
        result = {
            'mtcnn': mtcnn_result[0]['dominant_emotion'],
            'retinaface': retinaface_result[0]['dominant_emotion'],
            'mediapipe': media_pipe_result[0]['dominant_emotion']
        }
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, indent=2, default=str),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return json.dumps({'error': str(e)}, ensure_ascii=False), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)