import cv2
import torch
import base64
import threading
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Load your pre-trained models
object_detection_model = YOLO("C:/Users/kvsbb/Downloads/all.pt")  # Object detection model
pose_model = YOLO("C:/Users/kvsbb/Downloads/yolov8n-pose.pt")     # Pose detection model

# Global variables to manage video capture and detection state
cap = None
detection_thread = None
is_detecting = False
logs = []  # To store logs

# Function to generate frames with both object and pose detection
def generate_frames():
    global cap, is_detecting, logs
    while is_detecting:
        ret, frame = cap.read()
        if not ret:
            break

        # Start time for inference
        start_time = time.time()

        # Run object detection
        results_object = object_detection_model(frame)
        inference_time_object = (time.time() - start_time) * 1000  # Time in milliseconds
        annotated_frame = results_object[0].plot()

        # Capture and log object detection details
        if results_object[0].boxes.data.size(0) == 0:
            logs.append(f"No hard hats detected. Inference time: {inference_time_object:.2f}ms")
        else:
            for result in results_object[0].boxes.data:
                class_id = int(result[5].item())
                log_message = f"Object detected: Class {class_id}, Confidence {result[4]:.2f}"
                logs.append(log_message)

        # Run pose detection on the same frame
        start_time = time.time()
        results_pose = pose_model(frame)
        inference_time_pose = (time.time() - start_time) * 1000  # Time in milliseconds
        annotated_frame_pose = results_pose[0].plot()

        # Capture and log pose detection details
        if results_pose[0].keypoints:
            logs.append(f"Pose detected: {len(results_pose[0].keypoints)} keypoints. Inference time: {inference_time_pose:.2f}ms")
        else:
            logs.append(f"No poses detected. Inference time: {inference_time_pose:.2f}ms")

        # Combine both results (overlay pose detections on object detection frame)
        combined_frame = cv2.addWeighted(annotated_frame, 0.6, annotated_frame_pose, 0.4, 0)

        # Encode the frame as base64 to send to the frontend
        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # Send the frame to the client
        socketio.emit('video_frame', {'data': frame_data})

        # Send logs to the frontend (last 5 logs)
        socketio.emit('update_logs', {'logs': logs[-5:]})

        # Sleep to control the frame rate (approximately 25 FPS)
        socketio.sleep(0.04)

# Start detection (start capturing and streaming)
@app.route('/start_detection')
def start_detection():
    global cap, is_detecting, detection_thread
    if not is_detecting:
        cap = cv2.VideoCapture(0)  # Open the webcam
        is_detecting = True

        # Start the video streaming in a separate thread
        detection_thread = threading.Thread(target=generate_frames)
        detection_thread.daemon = True  # Ensure it closes when the app shuts down
        detection_thread.start()

        return 'Detection Started'
    return 'Detection already running'

# Stop detection (stop capturing and streaming)
@app.route('/stop_detection')
def stop_detection():
    global is_detecting, cap
    if is_detecting:
        is_detecting = False
        cap.release()  # Release the webcam
        return 'Detection Stopped'
    return 'Detection is not running'

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
