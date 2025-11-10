import os
import warnings
import streamlit as st
import cv2
import time
import threading
from collections import Counter
import builtins

# ───────────────────────────────
# Environment cleanup & warnings suppression
# ───────────────────────────────
os.environ.update({
    "PALIGEMMA_ENABLED": "False",
    "FLORENCE2_ENABLED": "False",
    "QWEN_2_5_ENABLED": "False",
    "CORE_MODEL_SAM_ENABLED": "False",
    "CORE_MODEL_SAM2_ENABLED": "False",
    "CORE_MODEL_CLIP_ENABLED": "False",
    "CORE_MODEL_GAZE_ENABLED": "False",
    "SMOLVLM2_ENABLED": "False",
    "DEPTH_ESTIMATION_ENABLED": "False",
    "MOONDREAM2_ENABLED": "False",
    "CORE_MODEL_TROCR_ENABLED": "False",
    "CORE_MODEL_GROUNDINGDINO_ENABLED": "False",
    "CORE_MODEL_YOLO_WORLD_ENABLED": "False",
    "CORE_MODEL_PE_ENABLED": "False",
})
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*use_container_width.*")

from inference import InferencePipeline

# --- Global alert state (thread-safe) -------------------------------------------------
# We'll capture terminal prints that announce a full bin and surface them in the UI.
alert_lock = threading.Lock()
ALERT_STATE = {"text": None, "time": 0.0}


# Monkey-patch the built-in print to catch alert messages like "{class} full! Empty now."
_orig_print = builtins.print

def _print_capture(*args, **kwargs):
    # call original print
    _orig_print(*args, **kwargs)
    try:
        sep = kwargs.get("sep", " ")
        s = sep.join(map(str, args))
        if "full! Empty now." in s:
            with alert_lock:
                ALERT_STATE["text"] = s
                ALERT_STATE["time"] = time.time()
    except Exception:
        pass


builtins.print = _print_capture

# ───────────────────────────────
# Streamlit page setup
# ───────────────────────────────
st.set_page_config(page_title="CleanSort CV", layout="centered")
st.title("♻️ CleanSort CV")

# Two-column layout: video (left) | insights (right)
# Reduce the left column relative width so the insights panel moves closer to the video.

# Left column placeholders

# Two columns: video (left) | insights (right)
col1, col2 = st.columns([2.5, 1])

# Left column placeholders
frame_placeholder = col1.empty()
status_placeholder = col1.empty()

# Alert placeholder sits below the video in the left column
alert_placeholder = col1.empty()

# Right column (Insights)
insights_container = col2.container()
insights_container.subheader("🧠 Insights")

# Emojis for each waste class
CLASS_EMOJIS = {
    "Recyclable waste": "🔁",
    "Kitchen waste": "🍎",
    "Other waste": "🗑️",
    "Hazardous waste": "☣️",
    "Glass": "🍾",
}
ALL_CLASSES = list(CLASS_EMOJIS.keys())

# Initialize global variables
_last_print_time = 0.0
frame_lock = threading.Lock()
latest_frame = None
latest_counts = Counter({cls: 0 for cls in ALL_CLASSES})

# Simple FPS tracker for overlay
_fps_last_time = None
_fps_smoothed = 0.0

# Persistent insights display placeholders
total_placeholder = insights_container.empty()
class_placeholders = {cls: insights_container.empty() for cls in ALL_CLASSES}

# ───────────────────────────────
# Inference callback function
# ───────────────────────────────
def my_sink(result, video_frame):
    global _last_print_time, latest_frame, latest_counts

    # Update frame
    if result.get("output_image"):
        output_img = result["output_image"].numpy_image
        frame_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        with frame_lock:
            latest_frame = frame_rgb

    # Throttle updates (2x per second)
    now = time.time()
    if now - _last_print_time < 0.5:
        return
    _last_print_time = now

    try:
        preds = result.get("predictions")
        detected_names = []

        if preds is not None and hasattr(preds, "data") and "class_name" in preds.data:
            arr = preds.data["class_name"]
            detected_names = list(map(str, arr.tolist())) if hasattr(arr, "tolist") else list(map(str, arr))

        class_counts = Counter(detected_names)
        latest_counts = Counter({cls: class_counts.get(cls, 0) for cls in ALL_CLASSES})

        # Print to terminal (clean summary)
        lines = ["──────────────"]
        lines.append(f"Total objects detected: {sum(latest_counts.values())}")
        for cls in ALL_CLASSES:
            emoji = CLASS_EMOJIS.get(cls, "❓")
            lines.append(f"  {emoji} {cls}: {latest_counts[cls]}")
        # Throttle updates (2x per second)
        if now - _last_print_time < 0.5:
            print("\n".join(lines))
        _last_print_time = now
        

    except Exception as e:
        print("Error parsing predictions:", e)


# ───────────────────────────────
# Start inference pipeline
# ───────────────────────────────
def start_pipeline():
    pipeline = InferencePipeline.init_with_workflow(
        api_key="XpA06GaUCSCLzmTK3oWx",
        workspace_name="experimentation-station",
        workflow_id="cleansort-cv-v2",
        video_reference=0,
        max_fps=30,
        on_prediction=my_sink,
    )
    pipeline.start()
    pipeline.join()

thread = threading.Thread(target=start_pipeline, daemon=True)
thread.start()

# ───────────────────────────────
# Streamlit render loop
# ───────────────────────────────
waiting_displayed = False

while True:
    # Update video
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is not None:
        status_placeholder.empty()
        # compute FPS
        now_f = time.time()
        if _fps_last_time is not None:
            dt = now_f - _fps_last_time
            if dt > 0:
                inst_fps = 1.0 / dt
                _fps_smoothed = (_fps_smoothed * 0.85) + (inst_fps * 0.15)
        else:
            _fps_smoothed = 0.0
        _fps_last_time = now_f

        # overlay FPS in top-right corner (small red text)
        try:
            disp = frame.copy()
            text = f"{_fps_smoothed:.1f} FPS"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = max(8, disp.shape[1] - tw - 8)
            y = 8 + th
            # image is RGB, so use RGB tuple for red
            color = (255, 0, 0)
            cv2.putText(disp, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
        except Exception:
            disp = frame

        frame_placeholder.image(disp, channels="RGB", width=1040)  # Slightly larger video feed
        waiting_displayed = False
    else:
        if not waiting_displayed:
            with status_placeholder.container():
                st.info("⏳ Waiting for camera feed...")
            waiting_displayed = True

    # Update static insights panel
    total_objects = sum(latest_counts.values())
    total_placeholder.markdown(f"**Total objects detected:** {total_objects}")
    for cls in ALL_CLASSES:
        emoji = CLASS_EMOJIS.get(cls, "❓")
        class_placeholders[cls].markdown(f"{emoji} **{cls}**: {latest_counts[cls]}")
    # Alert display logic: show last printed "full! Empty now." message as a pop-up for >=3s
    try:
        with alert_lock:
            alert_text = ALERT_STATE.get("text")
            alert_time = ALERT_STATE.get("time", 0.0)
        now_t = time.time()
        if alert_text and (now_t - alert_time) <= 3.0:
            # pastel red background with white text
            # prepend warning emoji
            warn_icon = "⚠️ "
            alert_html = f"<div style='background:#eb4034; color:#ffffff; padding:12px; border-radius:8px; border:1px solid #ffbcbc; min-width:220px; box-sizing:border-box;'><strong>{warn_icon}{alert_text}</strong></div>"
            alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
        else:
            alert_placeholder.empty()
            if alert_text and (now_t - alert_time) > 3.0:
                with alert_lock:
                    ALERT_STATE["text"] = None
                    ALERT_STATE["time"] = 0.0
    except Exception:
        try:
            alert_placeholder.empty()
        except Exception:
            pass
        

    time.sleep(0.03)  # ~30 FPS refresh
