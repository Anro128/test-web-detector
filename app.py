from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load YOLOv8 model sekali saat start
model = YOLO('./yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Resize untuk efisiensi
    max_dim = 640
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Inference YOLO
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    # Encode hasil ke base64 untuk dikirim balik
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': f'data:image/jpeg;base64,{encoded_image}'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
