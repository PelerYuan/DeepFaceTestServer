from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 用于将 numpy 类型转换为标准 JSON 可序列化格式
def convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    elif hasattr(obj, 'item'):  # 处理 float32 等 numpy scalar
        return obj.item()
    else:
        return obj

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == "":
        return jsonify({'error': 'Empty filename'}), 400

    # 保存临时文件
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    try:
        # DeepFace 情绪分析
        raw_result = DeepFace.analyze(
            img_path=filepath,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'  # 使用轻量后端加速处理
        )
        result = convert_ndarray(raw_result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    # 对外开放端口，支持 WSL / 局域网访问
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)