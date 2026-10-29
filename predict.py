import os
import re
import cv2
import joblib
import pickle
import numpy as np
try:
    from keras.models import load_model
except ImportError:
    load_model = None
    print("[CẢNH BÁO] Chưa cài TensorFlow/Keras -> tắt phân tích Video, chỉ chạy phân tích Caption")
from pyvi import ViTokenizer

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & TẢI MÔ HÌNH (violence.h5 & vihsd.pkl)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model_path(filename):
    """Tự động tìm file model ở thư mục gốc hoặc trong thư mục saved_models"""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "saved_models", filename),
        os.path.join(BASE_DIR, "models", filename)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

VIOLENCE_MODEL_PATH = find_model_path("violence_model.h5")
VIHSD_MODEL_PATH = find_model_path("vihsd_model.pkl")
VECTORIZER_PATH = find_model_path("vectorizer.pkl") # Dự phòng nếu bạn lưu TF-IDF riêng

# Tải mô hình Video (violence.h5)
vision_model = None
if load_model is not None and os.path.exists(VIOLENCE_MODEL_PATH):
    print(f"[INFO] Đang tải mô hình Video từ: {VIOLENCE_MODEL_PATH}")
    vision_model = load_model(VIOLENCE_MODEL_PATH, compile=False)
else:
    print(f"[CẢNH BÁO] Không tìm thấy file: {VIOLENCE_MODEL_PATH}")

# Tải mô hình NLP (vihsd.pkl)
nlp_bundle = None
vectorizer = None
if os.path.exists(VIHSD_MODEL_PATH):
    print(f"[INFO] Đang tải mô hình NLP từ: {VIHSD_MODEL_PATH}")
    try:
        nlp_bundle = joblib.load(VIHSD_MODEL_PATH)
    except Exception:
        with open(VIHSD_MODEL_PATH, "rb") as f:
            nlp_bundle = pickle.load(f)

    # Nếu bạn lưu rời file vectorizer.pkl bên ngoài
    if os.path.exists(VECTORIZER_PATH):
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception:
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
else:
    print(f"[CẢNH BÁO] Không tìm thấy file: {VIHSD_MODEL_PATH}")


# ==============================================================================
# 2. LUỒNG XỬ LÝ NGÔN NGỮ ĐỘC HẠI TIẾNG VIỆT (vihsd.pkl)
# ==============================================================================
def preprocess_vietnamese_text(text: str) -> str:
    """Làm sạch văn bản và tách từ ghép tiếng Việt bằng PyVi"""
    if not text or not str(text).strip():
        return ""
    text = str(text).lower()
    # Xóa đường dẫn URL, email, ký tự đặc biệt
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Tách từ tiếng Việt (ví dụ: "bạo lực" -> "bạo_lực")
    return ViTokenizer.tokenize(text)

def predict_text_risk(caption: str) -> float:
    """Dự đoán xác suất ngôn từ độc hại P_text trong khoảng [0.0, 1.0]"""
    cleaned_text = preprocess_vietnamese_text(caption)
    if not cleaned_text or nlp_bundle is None:
        return 0.0

    try:
        # Trường hợp 1: vihsd.pkl được lưu dạng dict {'vectorizer': ..., 'model': ...}
        if isinstance(nlp_bundle, dict):
            # vihsd_model.pkl hiện tại lưu dạng {'pipeline': Pipeline(TF-IDF + LogReg), 'threshold': 0.59, ...}
            if nlp_bundle.get("pipeline") is not None:
                return float( _get_prob(nlp_bundle["pipeline"], [cleaned_text]) )
            vec = nlp_bundle.get("vectorizer") or nlp_bundle.get("tfidf")
            clf = nlp_bundle.get("model") or nlp_bundle.get("classifier")
            features = vec.transform([cleaned_text])
            return float( _get_prob(clf, features) )

        # Trường hợp 2: vihsd.pkl được lưu dạng tuple/list (vectorizer, model)
        if isinstance(nlp_bundle, (tuple, list)) and len(nlp_bundle) == 2:
            vec, clf = nlp_bundle
            features = vec.transform([cleaned_text])
            return float( _get_prob(clf, features) )

        # Trường hợp 3: Có file vectorizer.pkl đi kèm bên ngoài
        if vectorizer is not None:
            features = vectorizer.transform([cleaned_text])
            return float( _get_prob(nlp_bundle, features) )

        # Trường hợp 4: vihsd.pkl là một Scikit-learn Pipeline hoàn chỉnh (đã gộp TF-IDF + Model)
        return float( _get_prob(nlp_bundle, [cleaned_text]) )

    except Exception as e:
        print(f"[LỖI DỰ ĐOÁN NLP]: {e}")
        return 0.0

def _get_prob(clf, X_input) -> float:
    """Lấy xác suất nhãn 1 (Vi phạm) từ bộ phân loại Scikit-learn"""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_input)[0]
        # Nếu phân loại nhị phân hoặc đa lớp (Clean=0, Offensive=1, Hate=2), lấy xác suất không phải nhãn 0
        return 1.0 - float(probs[0]) if len(probs) > 2 else float(probs[1])
    elif hasattr(clf, "decision_function"):
        score = float(clf.decision_function(X_input)[0])
        return float(1.0 / (1.0 + np.exp(-score))) # Sigmoid
    else:
        pred = int(clf.predict(X_input)[0])
        return 1.0 if pred > 0 else 0.0


# ==============================================================================
# 3. LUỒNG XỬ LÝ HÀNH VI BẠO LỰC VIDEO (violence.h5)
# ==============================================================================
def format_timestamp(seconds: int) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_window_frames(cap, start_frame: int, end_frame: int, num_frames=16, img_size=(128, 128)):
    """Trích xuất cố định 16 frames cách đều nhau trong cửa sổ 3 giây và chuẩn hóa về [0, 1]"""
    frames = []
    frame_indices = np.linspace(start_frame, max(start_frame, end_frame - 1), num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Nếu thiếu frame ở cuối video, bù bằng frame đen hoặc frame trước đó
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.float32))
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, img_size)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frames.append(frame_norm)

    return np.array(frames, dtype=np.float32) # Shape: (16, 128, 128, 3)

def predict_video_stream(video_path: str, window_sec=3, num_frames=16, img_size=(128, 128)):
    """Cắt video thành các phân đoạn 3 giây và dự đoán xác suất bạo lực trên từng đoạn"""
    if not video_path or not os.path.exists(video_path) or vision_model is None:
        return 0.0, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = max(1, int(np.ceil(total_frames / fps)))

    # Tự động đọc kích thước input từ model violence.h5 (nếu khác 16x128x128)
    try:
        input_shape = vision_model.input_shape
        if len(input_shape) == 5 and input_shape[1] is not None:
            num_frames = int(input_shape[1])
            img_size = (int(input_shape[3]), int(input_shape[2]))
    except Exception:
        pass

    segments = []
    max_p_visual = 0.0

    for start_sec in range(0, duration_sec, window_sec):
        end_sec = min(start_sec + window_sec, duration_sec)
        start_f = int(start_sec * fps)
        end_f = min(int(end_sec * fps), total_frames)

        if end_f <= start_f:
            continue

        # Trích xuất 16 khung hình cho phân đoạn hiện tại
        window_tensor = extract_window_frames(cap, start_f, end_f, num_frames=num_frames, img_size=img_size)
        batch_input = np.expand_dims(window_tensor, axis=0) # Shape: (1, 16, 128, 128, 3)

        # Suy luận với model violence.h5
        preds = vision_model.predict(batch_input, verbose=0)
        if preds.shape[-1] == 1:
            p_seg = float(preds[0][0])
        else:
            p_seg = float(preds[0][1]) # Nếu tầng cuối dùng Softmax 2 class [NonViolence, Violence]

        max_p_visual = max(max_p_visual, p_seg)
        risk_percent = round(p_seg * 100, 1)

        is_violent = p_seg >= 0.5
        segments.append({
            "start": start_sec,
            "end": end_sec,
            "time_label": f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}",
            "risk": risk_percent,
            "type": "VISION_VIOLENCE" if is_violent else "SAFE",
            "note": "Phát hiện hành vi xô xát, bạo lực thị giác" if is_violent else "Khung hình an toàn, chuyển động bình thường"
        })

    cap.release()
    return max_p_visual, segments


# ==============================================================================
# 4. HÀM HỢP NHẤT ĐA PHƯƠNG THỨC & TÍNH BRAND SAFETY SCORE
# ==============================================================================
def analyze_content(video_path: str = None, caption: str = "") -> dict:
    """
    Hàm chính được gọi từ server.py hoặc chạy kiểm tra độc lập.
    Trả về điểm Brand Safety Score (%) và chi tiết từng phân đoạn 3 giây.
    """
    # 1. Chạy luồng Video (violence.h5)
    p_visual, segments = predict_video_stream(video_path, window_sec=3)

    # 2. Chạy luồng Văn bản (vihsd.pkl)
    p_text = predict_text_risk(caption)

    # 3. Hợp nhất kết quả & tính Brand Safety Score (%)
    max_risk = max(p_visual, p_text)
    safety_score = round((1.0 - max_risk) * 100, 1)

    if safety_score >= 80:
        status = "SAFE"
        level_label = "AN TOÀN (Màu Xanh)"
    elif safety_score >= 50:
        status = "WARNING"
        level_label = "CẢNH BÁO (Màu Vàng)"
    else:
        status = "VIOLATION"
        level_label = "NGUY HIỂM / VI PHẠM (Màu Đỏ)"

    return {
        "safety_score": safety_score,
        "p_visual": round(p_visual * 100, 1),
        "p_text": round(p_text * 100, 1),
        "status": status,
        "level_label": level_label,
        "tokenized_caption": preprocess_vietnamese_text(caption),
        "segments": segments
    }


# ==============================================================================
# 5. CHẠY TEST TRỰC TIẾP TRONG TERMINAL (NẾU MUỐN KIỂM TRA NHANH)
# ==============================================================================
if __name__ == "__main__":
    # Thay đường dẫn bằng 1 file video bất kỳ trong tập test của bạn để kiểm thử
    sample_video = r"F:\SICT MediaTech Solutions Challenge\violence_data\test\Violence"
    test_video_file = None
    if os.path.exists(sample_video):
        files = [f for f in os.listdir(sample_video) if f.lower().endswith(('.mp4', '.avi'))]
        if files:
            test_video_file = os.path.join(sample_video, files[0])

    sample_caption = "Video đánh nhau gây rối trật tự công cộng cực kỳ bạo lực"

    print("\n--- ĐANG CHẠY KIỂM THỬ PREDICT.PY ---")
    result = analyze_content(video_path=test_video_file, caption=sample_caption)
    print(f"Video kiểm thử     : {test_video_file}")
    print(f"Caption sau PyVi   : {result['tokenized_caption']}")
    print(f"Rủi ro Hình ảnh    : {result['p_visual']}%")
    print(f"Rủi ro Ngôn từ     : {result['p_text']}%")
    print(f"Brand Safety Score : {result['safety_score']}% -> {result['level_label']}")
    print(f"Số phân đoạn 3 giây: {len(result['segments'])} đoạn")