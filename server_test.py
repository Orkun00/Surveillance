from flask import Flask, request, jsonify
import os
import cv2
import time
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

KNOWN_FACE_PATH = "orkun.jpeg"

@app.route("/upload", methods=["POST"])
def upload_image():
    overall_start = time.time()  # When server begins processing request

    # Get client-side timestamp if available
    client_send_time = request.form.get("t_send")
    if client_send_time:
        client_send_time = float(client_send_time)
        estimated_network_latency = overall_start - client_send_time
    else:
        estimated_network_latency = None

    if 'image' not in request.files:
        print("[ERROR] No image uploaded")
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    upload_start = time.time()
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    upload_end = time.time()
    save_time = upload_end - upload_start

    check_start = time.time()
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not saved"}), 500

    if cv2.imread(filepath) is None:
        return jsonify({"success": False, "error": "Saved image unreadable"}), 500
    check_time = time.time() - check_start

    try:
        verify_start = time.time()
        result = DeepFace.verify(
            img1_path=KNOWN_FACE_PATH,
            img2_path=filepath,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        verify_time = time.time() - verify_start
        total_time = time.time() - overall_start

        # Log everything
        print(f"[UPLOAD] File saved: {filename}")
        if estimated_network_latency is not None:
            print(f"  🌐 Network latency (client → server): {estimated_network_latency:.4f} sec")
        print(f"  💾 File save time:      {save_time:.4f} sec")
        print(f"  🔍 Image check time:    {check_time:.4f} sec")
        print(f"  🧠 DeepFace verify time:{verify_time:.4f} sec")
        print(f"  ⏱️ Total server time:    {total_time:.4f} sec")
        print(f"  ✅ Match result:         {result['verified']}, Distance: {result['distance']:.4f}")

        return jsonify({
            "success": True,
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "timing": {
                "client_to_server_latency": round(estimated_network_latency, 4) if estimated_network_latency else None,
                "file_save_time": round(save_time, 4),
                "image_check_time": round(check_time, 4),
                "deepface_verify_time": round(verify_time, 4),
                "total_server_time": round(total_time, 4)
            }
        })

    except Exception as e:
        print(f"[ERROR] DeepFace exception: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

