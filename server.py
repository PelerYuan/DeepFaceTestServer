from flask import Flask, request
from deepface import DeepFace
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

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
        raw_result = DeepFace.analyze(
            img_path=filepath,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='mtcnn'  # 使用更精准的人脸检测器
        )
        result = convert_ndarray(raw_result)
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