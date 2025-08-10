
from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import base64, cv2, os
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

app.config["AUDIO_FOLDER"] = "audio"
os.makedirs(app.config["AUDIO_FOLDER"], exist_ok=True)

# Load models
yolo = YOLO("yolo11s.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

def analyze_scene(frame):
    h, w = frame.shape[:2]
    results = yolo(frame, verbose=False)[0]
    desc = []

    for box in results.boxes:
        cls = int(box.cls)
        label = results.names[cls]
        x_center = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
        position = get_position(x_center, w)

        if label in ["person", "chair", "door", "car", "stair", "bicycle", "traffic light"]:
            desc.append(f"{label} on your {position}")

    # Caption with BLIP
    inputs = blip_processor(frame, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    full_description = f"{caption}. I see: {', '.join(desc)}."
    return full_description

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    description = analyze_scene(frame)
    return jsonify({'description': description})

if __name__ == '__main__':
    app.run(debug=True)

