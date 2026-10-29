# 🛡️ Brand Safety & Moderation Guard

**Brand Safety & Moderation Guard** là hệ thống kiểm duyệt nội dung đa phương thức (Multimodal Content Moderation) kết hợp giữa **Thị giác máy tính (Computer Vision)** và **Xử lý ngôn ngữ tự nhiên tiếng Việt (Vietnamese NLP)** nhằm phát hiện sớm hành vi bạo lực trong video và ngôn từ độc hại/thù ghét trong văn bản (caption), từ đó tính toán chỉ số an toàn thương hiệu (**Brand Safety Score**).

---

## ✨ Tính năng nổi bật

- **Phân tích Video theo phân đoạn (Video Violence Detection):**
  - Tự động chia nhỏ video thành các cửa sổ thời gian **3 giây**.
  - Trích xuất 16 khung hình (frames) cách đều nhau trên mỗi phân đoạn, chuẩn hóa kích thước `128x128` và đưa qua mô hình học sâu (`violence_model.h5`).
  - Khoanh vùng chính xác mốc thời gian (`MM:SS - MM:SS`) xuất hiện hành vi xô xát, bạo lực thị giác.
- **Kiểm duyệt Ngôn từ Độc hại Tiếng Việt (Vietnamese Hate Speech Detection):**
  - Làm sạch văn bản và tách từ ghép tiếng Việt tự động bằng thư viện **PyVi** (`ViTokenizer`).
  - Trích xuất đặc trưng TF-IDF kết hợp mô hình phân loại học máy (`vihsd_model.pkl`) để dự đoán xác suất vi phạm tiêu chuẩn cộng đồng.
- **Chấm điểm An toàn Thương hiệu (Brand Safety Score):**
  - Hợp nhất rủi ro từ cả hai luồng Hình ảnh ($P_{\text{visual}}$) và Ngôn từ ($P_{\text{text}}$):
    $$\text{Safety Score} = (1 - \max(P_{\text{visual}}, P_{\text{text}})) \times 100\%$$
  - Phân loại trực quan theo 3 cấp độ:
    - 🟢 **SAFE (`>= 80%`)**: Nội dung an toàn cho thương hiệu.
    - 🟡 **WARNING (`50% - 79%`)**: Cảnh báo rủi ro, cần kiểm duyệt viên xem xét.
    - 🔴 **VIOLATION (`< 50%`)**: Nguy hiểm / Vi phạm tiêu chuẩn.
- **Giao diện Dashboard trực quan:**
  - Xây dựng với giao diện Dark Mode hiện đại, hiển thị đồng hồ đo Brand Safety Score, kết quả tách từ PyVi và thanh timeline chi tiết cho từng phân đoạn 3 giây của video.

---

## 📂 Cấu trúc thư mục dự án

```text
SICT MediaTech Solutions Challenge/
├── brand_safety_moderation_guard_dashboard.html  # Giao diện người dùng (Frontend Dashboard)
├── server.py                                     # FastAPI Backend cung cấp REST API & giao diện
├── predict.py                                    # Luồng xử lý AI đa phương thức (Video + NLP)
├── split_data                                    # Script chia tập dữ liệu Train / Val / Test (70/20/10)
├── violence_model.h5                             # Trọng số mô hình nhận diện bạo lực Video (Keras/TensorFlow)
├── vihsd_model.pkl                               # Mô hình nhận diện ngôn từ độc hại tiếng Việt (Scikit-learn)
├── vihsd_data/                                   # Dữ liệu văn bản tiếng Việt (train.csv, val.csv, test.csv)
├── violence_data/                                # Dữ liệu video huấn luyện & kiểm thử (Violence / NonViolence)
└── temp_uploads/                                 # Thư mục lưu tạm video người dùng tải lên khi kiểm duyệt
```

---

## ⚙️ Yêu cầu hệ thống & Cài đặt

1. **Cài đặt Python** (khuyến nghị Python 3.10 – 3.13).
2. **Cài đặt các thư viện cần thiết:**

```bash
pip install fastapi uvicorn python-multipart tensorflow keras opencv-python scikit-learn joblib pyvi numpy pandas
```

---

## 🚀 Hướng dẫn khởi động dự án

**Bước 1:** Mở Terminal / PowerShell và di chuyển vào thư mục chứa mã nguồn (`server.py` và `predict.py`):

```powershell
cd "SICT MediaTech Solutions Challenge"
```

**Bước 2:** Khởi chạy server bằng câu lệnh:

```powershell
python -m uvicorn server:app --reload
```

*(Lưu ý: Lần đầu khởi động, TensorFlow sẽ mất khoảng 10–20 giây để nạp mô hình `violence_model.h5` và `vihsd_model.pkl` vào bộ nhớ cho đến khi hiện dòng chữ `Application startup complete.`)*

**Bước 3:** Mở trình duyệt web và truy cập vào địa chỉ:

👉 **http://127.0.0.1:8000**

---

## 🔌 API Endpoints

| Phương thức | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/` | Trả về giao diện Dashboard (`brand_safety_moderation_guard_dashboard.html`). |
| `POST` | `/api/moderate` | Nhận `video` (`UploadFile`) và `caption` (`Form`), trả về JSON kết quả phân tích đa phương thức. |

### Ví dụ JSON trả về từ `/api/moderate`:

```json
{
  "safety_score": 12.4,
  "p_visual": 87.6,
  "p_text": 15.2,
  "status": "VIOLATION",
  "level_label": "NGUY HIỂM / VI PHẠM (Màu Đỏ)",
  "tokenized_caption": "video đánh_nhau gây_rối trật_tự công_cộng",
  "segments": [
    {
      "start": 0,
      "end": 3,
      "time_label": "00:00 - 00:03",
      "risk": 87.6,
      "type": "VISION_VIOLENCE",
      "note": "Phát hiện hành vi xô xát, bạo lực thị giác"
    }
  ]
}
```
