from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

app = Flask(__name__)

# Load BLIP captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_scene(frame_bgr):
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    caption = describe_scene(frame)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
