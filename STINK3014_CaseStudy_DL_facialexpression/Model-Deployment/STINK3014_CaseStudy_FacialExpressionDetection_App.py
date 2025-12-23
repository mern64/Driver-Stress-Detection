import cv2
import tensorflow as tf
import numpy as np
import collections
import time
import sys
from datetime import datetime


# --- 1. MODEL SETUP ---
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    return model


try:
    print("Initializing System...")
    model = build_model()
    model.load_weights("stress_detector_cnn.h5")
    print("SUCCESS: AI Model Loaded. Starting Camera...")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load model weights.")
    print(f"Details: {e}")
    sys.exit()

# --- 2. LOGGING & BUFFERS ---
rage_buffer = collections.deque(maxlen=60)
score_history = collections.deque(maxlen=100)
event_log = collections.deque(maxlen=5)  # Keeps last 5 text events

# STATS VARIABLES
trip_start_time = time.time()
incident_count = 0
last_state = "OPTIMAL"
fps_start_time = time.time()
frame_count = 0
current_fps = 0
inference_time_ms = 0  # Time taken for AI to think

# --- 3. UI DESIGN SYSTEM ---
COLORS = {
    "background": (15, 15, 20),
    "cyan": (235, 206, 135),
    "safe": (100, 255, 100),
    "warn": (0, 165, 255),
    "danger": (0, 0, 255),
    "text_main": (240, 240, 240),
    "text_dim": (180, 180, 180),
    "black": (0, 0, 0)
}

RECOMMENDATIONS = [
    "PULL OVER SAFELY", "TAKE DEEP BREATHS", "PLAY CALMING MUSIC", "DRINK WATER"
]


def log_event(msg):
    """Adds a timestamped message to the scrolling log"""
    t_str = datetime.now().strftime("%H:%M:%S")
    # Avoid duplicate spam
    if not event_log or event_log[-1][1] != msg:
        event_log.append((t_str, msg))


def get_ux_state(score):
    if score < 0.3:
        return "OPTIMAL", COLORS["safe"], "SYSTEM NORMAL", ""
    elif score < 0.7:
        return "CAUTION", COLORS["warn"], "STRESS RISING", "FOCUS ON ROAD"
    else:
        rec = RECOMMENDATIONS[int(time.time()) % len(RECOMMENDATIONS)]
        return "DANGER", COLORS["danger"], "ROAD RAGE DETECTED", rec


def draw_tech_box(img, x, y, w, h, color, title=None, fill=False):
    sub_img = img[y:y + h, x:x + w]
    if fill:
        rect = np.zeros(sub_img.shape, dtype=np.uint8)
        rect[:] = color
        res = cv2.addWeighted(sub_img, 0.3, rect, 0.7, 1.0)
    else:
        dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        dark_rect[:] = (10, 10, 10)
        res = cv2.addWeighted(sub_img, 0.6, dark_rect, 0.4, 1.0)
    img[y:y + h, x:x + w] = res

    thk = 3 if fill else 1
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Corner Accents
    line_len = 10
    cv2.line(img, (x, y), (x + line_len, y), color, thk)
    cv2.line(img, (x, y), (x, y + line_len), color, thk)
    cv2.line(img, (x + w, y), (x + w - line_len, y), color, thk)
    cv2.line(img, (x + w, y), (x + w, y + line_len), color, thk)
    cv2.line(img, (x, y + h), (x + line_len, y + h), color, thk)
    cv2.line(img, (x, y + h), (x, y + h - line_len), color, thk)
    cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thk)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thk)

    if title:
        text_col = COLORS["black"] if fill else COLORS["cyan"]
        cv2.putText(img, title, (x + 10, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_col, 1, cv2.LINE_AA)


def draw_live_graph(img, x, y, w, h, data, color):
    draw_tech_box(img, x, y, w, h, color, "STRESS HISTORY")
    thresh_danger_y = int(y + h - (0.7 * h))
    cv2.line(img, (x, thresh_danger_y), (x + w, thresh_danger_y), (50, 50, 150), 1)

    if len(data) < 2: return
    step_x = w / (data.maxlen - 1)
    pts = []
    for i, val in enumerate(data):
        px = int(x + (i * step_x))
        val_clamped = max(0.0, min(1.0, val))
        py = int(y + h - (val_clamped * (h - 20)) - 10)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret: break
    h_res, w_res, _ = frame.shape
    frame = cv2.flip(frame, 1)

    # Calculate FPS
    frame_count += 1
    if time.time() - fps_start_time > 1:
        current_fps = frame_count
        frame_count = 0
        fps_start_time = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_prediction = 0
    raw_confidence = 0.0
    processed_face = None  # Variable to hold the face image for UI

    # --- PROCESSING ---
    if len(faces) == 0:
        log_event("No Face Detected")

    for (x, y, wf, hf) in faces:
        # Pre-processing
        roi_original = gray[y:y + hf, x:x + wf]
        roi = cv2.resize(roi_original, (48, 48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        # Save visualization of what AI sees
        processed_face = cv2.resize(roi_original, (100, 100))

        # Inference Timing
        t_start = time.time()
        preds = model.predict(roi, verbose=0)[0]
        t_end = time.time()
        inference_time_ms = (t_end - t_start) * 1000

        current_prediction = np.argmax(preds)
        raw_confidence = np.max(preds)

        box_color = COLORS["safe"] if current_prediction == 0 else COLORS["danger"]
        cv2.rectangle(frame, (x, y), (x + wf, y + hf), box_color, 2)
        cv2.putText(frame, f"{raw_confidence * 100:.0f}%", (x + wf + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    box_color, 1)

    rage_buffer.append(current_prediction)
    score = np.mean(rage_buffer) if rage_buffer else 0
    score_history.append(score)

    status_text, theme_color, alert_msg, rec_msg = get_ux_state(score)

    # Log significant state changes
    if status_text != last_state:
        log_event(f"State: {status_text}")
        if status_text == "DANGER":
            incident_count += 1
    last_state = status_text

    # --- UI RENDERING LAYER ---

    # 1. WARNING BORDER
    if score >= 0.7:
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w_res, h_res), COLORS["danger"], 20)

    # 2. LEFT COLUMN: SYSTEM STATUS & TELEMETRY
    draw_tech_box(frame, 20, 20, 220, 90, theme_color, "SYSTEM STATUS")
    cv2.putText(frame, status_text, (35, 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, theme_color, 2, cv2.LINE_AA)

    elapsed = int(time.time() - trip_start_time)
    min_s, sec_s = divmod(elapsed, 60)

    draw_tech_box(frame, 20, 120, 200, 140, COLORS["text_dim"], "MISSION DATA")
    # Trip Time
    cv2.putText(frame, "T-TIME:", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_dim"], 1)
    cv2.putText(frame, f"{min_s:02}:{sec_s:02}", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_main"], 1)
    # Incidents
    cv2.putText(frame, "ALERTS:", (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_dim"], 1)
    cv2.putText(frame, str(incident_count), (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["warn"], 1)
    # Avg Stress
    avg_s = np.mean(score_history) if score_history else 0
    cv2.putText(frame, "AVG LOAD:", (35, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_dim"], 1)
    cv2.putText(frame, f"{avg_s * 100:.0f}%", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_main"], 1)
    # Latency
    cv2.putText(frame, "LATENCY:", (35, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_dim"], 1)
    cv2.putText(frame, f"{inference_time_ms:.1f}ms", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["cyan"], 1)

    # 3. BOTTOM LEFT: EVENT LOG
    draw_tech_box(frame, 20, h_res - 140, 250, 120, COLORS["text_dim"], "SYSTEM LOG")
    for i, (t, msg) in enumerate(event_log):
        y_pos = h_res - 110 + (i * 20)
        # Latest event is brighter
        c = COLORS["text_main"] if i == len(event_log) - 1 else COLORS["text_dim"]
        cv2.putText(frame, f"[{t}] {msg}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    # 4. TOP RIGHT: PERFORMANCE & NEURAL VIEW
    # FPS
    cv2.putText(frame, f"FPS: {current_fps}", (w_res - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["safe"], 1)

    # Neural Vision Window (Show what the AI sees)
    if processed_face is not None:
        # Convert grayscale face back to BGR for display
        face_display = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)
        # Draw box for it
        vis_x, vis_y = w_res - 120, 50
        draw_tech_box(frame, vis_x - 10, vis_y - 20, 120, 130, COLORS["cyan"], "NEURAL INPUT")
        # Overlay the image
        frame[vis_y:vis_y + 100, vis_x:vis_x + 100] = face_display

    # 5. BOTTOM RIGHT: LIVE GRAPH
    draw_live_graph(frame, w_res - 240, h_res - 120, 220, 100, score_history, theme_color)

    # 6. CENTER ALERT SYSTEM
    if score >= 0.3:
        cw, ch = 500, 150
        cx, cy = (w_res - cw) // 2, (h_res - ch) // 2
        is_danger = score >= 0.7
        box_color = COLORS["danger"] if is_danger else COLORS["warn"]
        fill_box = is_danger and (int(time.time() * 5) % 2 == 0)

        draw_tech_box(frame, cx, cy, cw, ch, box_color, "ALERT SYSTEM", fill=fill_box)
        text_color = COLORS["black"] if fill_box else COLORS["text_main"]
        cv2.putText(frame, alert_msg, (cx + 20, cy + 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, text_color, 3)
        if rec_msg:
            cv2.putText(frame, f"REC: {rec_msg}", (cx + 20, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # 7. BOTTOM CENTER: STABILITY GAUGE
    bar_w = 300
    start_x = (w_res - bar_w) // 2
    bar_y = h_res - 50
    draw_tech_box(frame, start_x - 10, bar_y - 30, bar_w + 20, 50, COLORS["text_dim"])
    segments = 20
    segment_w = bar_w // segments
    filled_segments = int((1 - score) * segments)
    for i in range(segments):
        sx = start_x + (i * segment_w)
        if i < filled_segments:
            c = COLORS["safe"] if i > segments * 0.6 else COLORS["danger"]
            cv2.rectangle(frame, (sx, bar_y - 10), (sx + segment_w - 2, bar_y + 10), c, -1)

    cv2.imshow("STINK3014 - Smart Drive Management System HUD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()