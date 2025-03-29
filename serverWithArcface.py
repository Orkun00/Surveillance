import time

from flask import Flask, request, jsonify
import os
import cv2
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

KNOWN_FACE_PATH = ".jpg"

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Check file saved and is readable
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not saved"}), 500

    if cv2.imread(filepath) is None:
        return jsonify({"success": False, "error": "Saved image unreadable"}), 500

    try:
        start_time = time.time()

        result = DeepFace.verify(
            img1_path=KNOWN_FACE_PATH,
            img2_path=filepath,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )

        elapsed = time.time() - start_time

        return jsonify({
            "success": True,
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "processing_time_sec": round(elapsed, 3)
        })


    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
