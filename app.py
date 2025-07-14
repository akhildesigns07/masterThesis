# from flask import Flask, render_template, request, jsonify, send_file
# import cv2
# import numpy as np
# import base64
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from ultralytics import YOLO
# from gtts import gTTS
# import torch
# import os

# app = Flask(__name__)

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# yolo_model = YOLO("yolov8n.pt")

# def get_position(x_center, frame_width):
#     if x_center < frame_width / 3:
#         return "left"
#     elif x_center < 2 * frame_width / 3:
#         return "center"
#     else:
#         return "right"

# def analyze_frame(frame, results):
#     h, w = frame.shape[:2]
#     detections = results[0]
#     desc = []

#     environment = "indoor" if any(cls in detections.names and detections.names[cls] in ['chair', 'couch', 'tv', 'door'] for cls in detections.boxes.cls.tolist()) else "outdoor"

#     for box in detections.boxes:
#         cls = int(box.cls)
#         label = detections.names[cls]
#         x_center = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
#         position = get_position(x_center, w)

#         if label in ["person", "chair", "door", "car", "stair", "bicycle", "traffic light"]:
#             desc.append(f"{label} on your {position}")

#     # Use BLIP for more descriptive understanding
#     inputs = processor(frame, return_tensors="pt")
#     out = blip_model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)

#     full_description = f"{'Indoor' if environment=='indoor' else 'Outdoor'} scene. {caption}. I see: {', '.join(desc)}."
#     return full_description

# def describe_scene(frame_bgr):
#     image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
#     inputs = processor(images=image, return_tensors="pt")
#     out = blip_model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption

# def spatial_positions(frame_bgr):
#     results = yolo_model.predict(frame_bgr, verbose=False)
#     h, w = frame_bgr.shape[:2]
#     positions = []

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             label = yolo_model.names[int(box.cls[0])]
#             center_x = (x1 + x2) / 2

#             if center_x < w / 3:
#                 pos = "on your left"
#             elif center_x < 2 * w / 3:
#                 pos = "in front of you"
#             else:
#                 pos = "on your right"

#             positions.append(f"{label} {pos}")

#     if positions:
#         return " and a ".join(positions)
#     else:
#         return "no significant objects detected"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     data = request.json['image']
#     img_data = base64.b64decode(data.split(',')[1])
#     nparr = np.frombuffer(img_data, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     caption = describe_scene(frame)
#     location_info = spatial_positions(frame)
#     full_text = f"{caption}. You can see a {location_info}."

#     # Generate TTS
#     tts = gTTS(full_text)
#     tts_path = "static/tts_output.mp3"
#     tts.save(tts_path)

#     return jsonify({'caption': full_text, 'audio_url': tts_path})

# @app.route('/tts')
# def tts():
#     return send_file("static/tts_output.mp3", mimetype='audio/mpeg')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")

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

   # environment = "indoor" if any(results.names[int(cls)] in ['chair', 'door', 'tv', 'couch']
    #                              for cls in results.boxes.cls.tolist()) else "outdoor"

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

    #full_description = f"{environment.capitalize()} scene. {caption}. I see: {', '.join(desc)}."

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
