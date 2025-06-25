from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import os

app = Flask(__name__)

# Load model YOLOv8
model = YOLO('./yolov8n.pt')  # ganti dengan path model kamu

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inference YOLO
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    # Convert to base64 for return
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{encoded_image}'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
