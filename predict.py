import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import cv2
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from pyvi import ViTokenizer

# ==============================================================================
# 0. KHỞI TẠO EASYOCR CHO TIẾNG VIỆT
# ==============================================================================
try:
    import easyocr
    print("[INFO] Đang khởi tạo EasyOCR cho tiếng Việt...")
    ocr_reader = easyocr.Reader(['vi'], gpu=False, verbose=False)
    print("[INFO] Đã khởi tạo EasyOCR thành công!")
except Exception as e:
    ocr_reader = None
    print(f"[CẢNH BÁO] Không thể khởi tạo EasyOCR: {e}")

def extract_subtitle_from_frame(frame_bgr) -> str:
    """
    Trích xuất chữ / phụ đề trên khung hình video bằng EasyOCR kết hợp xử lý ảnh OpenCV.
    """
    if ocr_reader is None or frame_bgr is None:
        return ""
    try:
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        # 1. Chuyển sang ảnh xám để loại bỏ nhiễu màu phức tạp
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Phóng to gấp đôi để EasyOCR dễ dàng bắt nét các dòng phụ đề nhỏ
        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 3. Gọi EasyOCR đọc văn bản
        results = ocr_reader.readtext(resized, detail=0, paragraph=True)
        
        detected_text = " ".join(results).strip()
        if detected_text:
            print(f"[EASYOCR ĐỌC ĐƯỢC CHỮ]: {detected_text}")
        return detected_text

    except Exception as e:
        print(f"[LỖI EASYOCR CHI TIẾT]: {e}")
        return ""


# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & TẢI MÔ HÌNH (violence.h5 & vihsd.pkl)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model_path(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    candidates = []
    for filename in filenames:
        candidates.extend([
            os.path.join(BASE_DIR, filename),
            os.path.join(BASE_DIR, "saved_models", filename),
            os.path.join(BASE_DIR, "models", filename),
            os.path.join(os.path.dirname(BASE_DIR), "models", filename),
            os.path.join(os.path.dirname(BASE_DIR), filename),
            os.path.join(os.getcwd(), "models", filename),
            os.path.join(os.getcwd(), filename),
        ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

VIOLENCE_MODEL_PATH = find_model_path(["violence_model.h5", "violence.h5"])
VIHSD_MODEL_PATH = find_model_path(["vihsd_model.pkl", "vihsd.pkl"])
VECTORIZER_PATH = find_model_path(["vectorizer.pkl"])

def build_violence_detector(num_frames=16, img_size=128):
    try:
        from tensorflow.keras import layers, models, applications
    except Exception:
        import keras
        from keras import layers, models, applications

    base_cnn = applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        pooling="avg",
        weights=None
    )
    base_cnn.trainable = False

    inp = layers.Input(shape=(num_frames, img_size, img_size, 3), name="video_input")
    x = layers.TimeDistributed(base_cnn, name="frame_features")(inp)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        name="bi_lstm"
    )(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    out = layers.Dense(1, activation="sigmoid", name="violence_prob")(x)

    return models.Model(inputs=inp, outputs=out, name="violence_detector")

vision_model = None
if os.path.exists(VIOLENCE_MODEL_PATH):
    print(f"[INFO] Đang tải mô hình Video từ: {VIOLENCE_MODEL_PATH}")
    try:
        vision_model = load_model(VIOLENCE_MODEL_PATH, compile=False)
    except Exception as e:
        try:
            vision_model = build_violence_detector()
            vision_model.load_weights(VIOLENCE_MODEL_PATH)
        except Exception as err:
            print(f"[LỖI] Không thể nạp weights mô hình Video: {err}")

nlp_bundle = None
vectorizer = None
if os.path.exists(VIHSD_MODEL_PATH):
    print(f"[INFO] Đang tải mô hình NLP từ: {VIHSD_MODEL_PATH}")
    try:
        nlp_bundle = joblib.load(VIHSD_MODEL_PATH)
    except Exception:
        with open(VIHSD_MODEL_PATH, "rb") as f:
            nlp_bundle = pickle.load(f)

    if os.path.exists(VECTORIZER_PATH):
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception:
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)


# ==============================================================================
# 2. XỬ LÝ NGÔN NGỮ TIẾNG VIỆT (vihsd.pkl)
# ==============================================================================
def preprocess_vietnamese_text(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổổơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ViTokenizer.tokenize(text)

def predict_text_risk(text: str) -> float:
    cleaned_text = preprocess_vietnamese_text(text)
    if not cleaned_text or nlp_bundle is None:
        return 0.0

    try:
        if isinstance(nlp_bundle, dict):
            if "pipeline" in nlp_bundle:
                return float(_get_prob(nlp_bundle["pipeline"], [cleaned_text]))
            vec = nlp_bundle.get("vectorizer") or nlp_bundle.get("tfidf")
            clf = nlp_bundle.get("model") or nlp_bundle.get("classifier")
            if vec is not None and clf is not None:
                return float(_get_prob(clf, vec.transform([cleaned_text])))

        if isinstance(nlp_bundle, (tuple, list)) and len(nlp_bundle) == 2:
            vec, clf = nlp_bundle
            return float(_get_prob(clf, vec.transform([cleaned_text])))

        if vectorizer is not None:
            return float(_get_prob(nlp_bundle, vectorizer.transform([cleaned_text])))

        return float(_get_prob(nlp_bundle, [cleaned_text]))
    except Exception as e:
        print(f"[LỖI DỰ ĐOÁN NLP]: {e}")
        return 0.0

def _get_prob(clf, X_input) -> float:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_input)[0]
        return 1.0 - float(probs[0]) if len(probs) > 2 else float(probs[1])
    elif hasattr(clf, "decision_function"):
        score = float(clf.decision_function(X_input)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    else:
        pred = int(clf.predict(X_input)[0])
        return 1.0 if pred > 0 else 0.0


# ==============================================================================
# 3. XỬ LÝ VIDEO & HỢP NHẤT ĐA PHƯƠNG THỨC
# ==============================================================================
def format_timestamp(seconds: int) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_window_data(cap, start_frame: int, end_frame: int, num_frames=16, img_size=(128, 128)):
    vision_frames = []
    keyframe_bgr = None
    frame_indices = np.linspace(start_frame, max(start_frame, end_frame - 1), num_frames, dtype=int)
    mid_idx = num_frames // 2

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            if len(vision_frames) > 0:
                vision_frames.append(vision_frames[-1])
            else:
                vision_frames.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.float32))
            continue

        if i == mid_idx or keyframe_bgr is None:
            keyframe_bgr = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, img_size)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        vision_frames.append(frame_norm)

    return np.array(vision_frames, dtype=np.float32), keyframe_bgr

def process_video_dual_stream(video_path: str, window_sec=3, num_frames=16, img_size=(128, 128)):
    if not video_path or not os.path.exists(video_path):
        return 0.0, 0.0, [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0.0, [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = max(1, int(np.ceil(total_frames / fps)))

    batch_vision_tensors = []
    keyframes_for_ocr = []
    seg_times = []

    for start_sec in range(0, duration_sec, window_sec):
        end_sec = min(start_sec + window_sec, duration_sec)
        start_f = int(start_sec * fps)
        end_f = min(int(end_sec * fps), total_frames)
        if end_f <= start_f:
            continue

        v_tensor, k_frame = extract_window_data(cap, start_f, end_f, num_frames=num_frames, img_size=img_size)
        batch_vision_tensors.append(v_tensor)
        keyframes_for_ocr.append(k_frame)
        seg_times.append((start_sec, end_sec))

    cap.release()

    if not seg_times:
        return 0.0, 0.0, [], []

    vision_probs = [0.0] * len(seg_times)
    if vision_model is not None and batch_vision_tensors:
        batch_input = np.array(batch_vision_tensors, dtype=np.float32)
        preds_all = vision_model.predict(batch_input, verbose=0)
        for i in range(len(seg_times)):
            preds = preds_all[i]
            if preds.shape[-1] == 1 or len(preds.shape) == 0:
                vision_probs[i] = float(preds[0] if len(preds.shape) > 0 else preds)
            else:
                vision_probs[i] = float(preds[1])

    segments = []
    detected_subtitles = []
    max_p_visual = 0.0
    max_p_video_text = 0.0

    # QUÉT OCR THÔNG MINH (SIÊU TỐC): Chỉ chạy OCR trên khung hình giữa video để bắt trọn phụ đề mà không làm chậm máy
    global_sub_text = ""
    if keyframes_for_ocr:
        mid_idx = len(keyframes_for_ocr) // 2
        global_sub_text = extract_subtitle_from_frame(keyframes_for_ocr[mid_idx])
        if global_sub_text:
            max_p_video_text = predict_text_risk(global_sub_text)
            detected_subtitles.append(f"[Phụ đề video] {global_sub_text}")

    for i, (start_sec, end_sec) in enumerate(seg_times):
        p_vis = vision_probs[i]
        max_p_visual = max(max_p_visual, p_vis)

        p_sub = max_p_video_text if global_sub_text else 0.0
        seg_risk = round(max(p_vis, p_sub) * 100, 1)
        vis_violent = p_vis >= 0.5
        sub_toxic = p_sub >= 0.5

        if vis_violent and sub_toxic:
            seg_type = "BOTH_VIOLATION"
            note = f"Bạo lực & Phụ đề độc hại: '{global_sub_text}'"
        elif vis_violent:
            seg_type = "VISION_VIOLENCE"
            note = f"Phát hiện hành vi xô xát, bạo lực thị giác ({round(p_vis*100,1)}%)"
        elif sub_toxic:
            seg_type = "TEXT_VIOLATION"
            note = f"Phát hiện phụ đề độc hại: '{global_sub_text}'"
        else:
            seg_type = "SAFE"
            note = "An toàn" + (f" (Chữ trên video: '{global_sub_text}')" if global_sub_text else "")

        segments.append({
            "start": start_sec,
            "end": end_sec,
            "time_label": f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}",
            "risk": seg_risk,
            "p_visual_seg": round(p_vis * 100, 1),
            "p_text_seg": round(p_sub * 100, 1),
            "subtitle": global_sub_text,
            "type": seg_type,
            "note": note
        })

    return max_p_visual, max_p_video_text, segments, detected_subtitles

def predict_video_stream(video_path: str, window_sec=3, num_frames=16, img_size=(128, 128)):
    """
    Hàm tương thích ngược cho bộ kiểm thử test_suite.py.
    Trả về (max_p_visual, segments).
    """
    max_p_visual, _, segments, _ = process_video_dual_stream(
        video_path, window_sec=window_sec, num_frames=num_frames, img_size=img_size
    )
    return max_p_visual, segments

def analyze_content(video_path: str = None, caption: str = "") -> dict:
    p_visual, p_video_text, segments, video_subtitles = process_video_dual_stream(video_path, window_sec=3)
    p_caption_text = predict_text_risk(caption)
    p_text = max(p_video_text, p_caption_text)

    combined_raw_text = " | ".join(filter(None, [caption] + video_subtitles))
    tokenized_output = preprocess_vietnamese_text(combined_raw_text)

    max_risk = max(p_visual, p_text)
    safety_score = round((1.0 - max_risk) * 100, 1)

    if safety_score >= 80:
        status, level_label = "SAFE", "AN TOÀN (Màu Xanh)"
    elif safety_score >= 50:
        status, level_label = "WARNING", "CẢNH BÁO (Màu Vàng)"
    else:
        status, level_label = "VIOLATION", "NGUY HIỂM / VI PHẠM (Màu Đỏ)"

    return {
        "safety_score": safety_score,
        "p_visual": round(p_visual * 100, 1),
        "p_text": round(p_text * 100, 1),
        "status": status,
        "level_label": level_label,
        "tokenized_caption": tokenized_output,
        "detected_subtitles": video_subtitles,
        "segments": segments
    }