from flask import Flask, request
import numpy as np
import cv2
import os
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    t_recv = time.time()

    # Decode image
    image_data = request.files['image'].read()
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Save image
    os.makedirs("received", exist_ok=True)
    filename = f"received_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join("received", filename)
    cv2.imwrite(save_path, img)
    t_save_done = time.time()

    # Pull RPi timestamps
    try:
        t_capture_start = float(request.form.get("t_capture_start", 0))
        t_capture_end = float(request.form.get("t_capture_end", 0))
        t_encode_start = float(request.form.get("t_encode_start", 0))
        t_encode_end = float(request.form.get("t_encode_end", 0))
        t_send = float(request.form.get("t_send", 0))

        # Calculate durations
        times = {
            "📸 Capture Time": t_capture_end - t_capture_start,
            "🧃 Encode Time": t_encode_end - t_encode_start,
            "📤 Network Delay": t_recv - t_send,
            "💾 Save Time": t_save_done - t_recv,
            "⏱️ Total (Capture → Save)": t_save_done - t_capture_start
        }

        print("🕒 Timing Breakdown:")
        for k, v in times.items():
            print(f"{k}: {v:.3f} sec")

        return f"Success. Total: {times['⏱️ Total (Capture → Save)']:.3f}s", 200

    except Exception as e:
        return f"Error parsing timestamps: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
