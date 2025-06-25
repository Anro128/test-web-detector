import os
import cv2
from ultralytics import YOLO
from flask import Flask, Response, render_template

# Inisialisasi Flask
app = Flask(__name__)

# Load YOLO model (ganti path sesuai dengan lokasi model kamu)
model = YOLO('D:/project/sampah/best.pt')

# Generator frame dari webcam
def generate_frames():
    cap = cv2.VideoCapture(0)  # Webcam lokal
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi objek
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Encode ke JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Stream sebagai multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route utama
@app.route('/')
def index():
    return render_template('index.html')

# Route video stream
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
