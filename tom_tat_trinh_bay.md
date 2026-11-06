# 🛡️ SICT MediaTech — Tóm tắt Trình bày Đề tài

> Hội đồng thi · Ngày 07/11/2026

---

## 1. 🎯 Vấn đề & Động lực

**Bối cảnh thực tế:**
- Nền tảng video ngắn (TikTok, Reels, Shorts) bùng nổ → nội dung độc hại xuất hiện ngày càng tinh vi.
- Nội dung vi phạm thường tấn công **đồng thời qua 3 kênh**:
  1. **Hình ảnh chuyển động** — cảnh xô xát, bạo lực thị giác.
  2. **Phụ đề chèn trên video** — thoại nhân vật chứa ngôn từ thù địch.
  3. **Caption / bình luận** — văn bản lách luật kiểm duyệt.
- Hệ thống kiểm duyệt chỉ dựa vào 1 chiều dữ liệu **bỏ sót nhiều trường hợp vi phạm**.

**Mục tiêu đề tài:**
Xây dựng hệ thống **Brand Safety Score** kiểm duyệt nội dung đa phương thức, giúp nhà quảng cáo biết video nào an toàn để chạy quảng cáo.

---

## 2. 🏗️ Kiến trúc tổng thể — 3 luồng AI + Fusion

```
           VIDEO + CAPTION
                │
      ┌─────────┼──────────┐
      ▼         ▼          ▼
 [Luồng 1]  [Luồng 3]  [Luồng 2]
 Vision     Subtitle    NLP
 Stream     OCR         Stream
      │         │          │
   P_visual  P_subtitle  P_caption
      │         └────┬─────┘
      │        P_text = max(P_sub, P_cap)
      │              │
      └──────┬────────┘
       Max Risk = max(P_vis, P_text)
             │
   Brand Safety Score = (1 - Max Risk) × 100
```

| Luồng | Công nghệ | Mô hình / Module |
|---|---|---|
| **Vision** | MobileNetV2 + BiLSTM | `models/violence_model.h5` |
| **Subtitle OCR** | EasyOCR + Watermark Filter | `UI/predict.py` |
| **NLP Caption** | TF-IDF + Ensemble Classifier | `models/vihsd_model.pkl` |
| **Fusion** | Max-Risk Principle | `UI/predict.py` |

---

## 3. 📊 Thang điểm Brand Safety Score

| Score | Nhãn | Màu sắc | Hành động |
|:---:|:---:|:---:|---|
| 80 – 100% | AN TOÀN | 🟢 Xanh | Phân phối quảng cáo bình thường |
| 50 – 79.9% | CẢNH BÁO | 🟡 Vàng | Đưa vào hàng đợi kiểm duyệt thủ công |
| < 50% | VI PHẠM | 🔴 Đỏ | Chặn hiển thị ngay lập tức |

---

## 4. 🎥 PHẦN CỦA BẠN — Huấn luyện Mô hình Nhận diện Bạo lực Video

### 4.1 Tại sao dùng MobileNetV2 + BiLSTM?

| Thành phần | Vai trò | Lý do chọn |
|---|---|---|
| **MobileNetV2** | Trích xuất đặc trưng không gian (Spatial Feature Extractor) | Nhẹ, nhanh, pre-trained ImageNet → Transfer Learning hiệu quả |
| **Bidirectional LSTM** | Học quan hệ thời gian giữa các frame | Phân tích hành động theo **2 chiều thời gian** (trước → sau, sau → trước) |
| **TimeDistributed** | Áp dụng MobileNetV2 độc lập cho từng frame trong clip | Giữ nguyên chuỗi thời gian để LSTM phân tích |

### 4.2 Kiến trúc mạng chi tiết

```
Input (B, 16, 128, 128, 3)   ← Mỗi video clip: 16 frame, 128×128 px, RGB
  │
  ↓ Rescaling(scale=2.0, offset=-1.0)
     → Chuẩn hóa pixel từ [0, 1] về [-1, 1] theo chuẩn MobileNetV2
  │
  ↓ TimeDistributed(MobileNetV2)
     → Mỗi frame trở thành vector đặc trưng 1280 chiều
     → (B, 16, 1280)
  │
  ↓ TimeDistributed(Dropout(0.3) + BatchNorm)
     → Chống overfitting, ổn định phân phối
  │
  ↓ Bidirectional(LSTM(256, return_sequences=True))
     → Học ngữ cảnh thời gian theo 2 hướng
     → (B, 16, 512)
  │
  ↓ Dropout(0.3)
  │
  ↓ LSTM(128, return_sequences=False)
     → Tổng hợp toàn bộ chuỗi thành 1 vector
     → (B, 128)
  │
  ↓ Dense(128, relu) → Dropout(0.4)
  │
  ↓ Dense(1, sigmoid)   ← P_visual ∈ [0.0, 1.0]
```

### 4.3 Dữ liệu huấn luyện

**Cấu trúc thư mục dữ liệu video:**
```
data/violence_data/
├── train/
│   ├── Violence/      ← V_1.mp4 ... V_xxx.mp4
│   └── NonViolence/   ← NV_1.mp4 ... NV_xxx.mp4
├── val/
│   ├── Violence/
│   └── NonViolence/
└── test/
    ├── Violence/
    └── NonViolence/
```

**Tiền xử lý (`preprocessing/video_preprocess.py`):**
1. Mỗi video → trích **16 frames phân bố đều** theo thời lượng.
2. Resize về **128 × 128 px**.
3. Chuẩn hóa pixel về `[0.0, 1.0]` kiểu `float32`.
4. Lưu thành file npy:
   - `train_X.npy` shape `(N, 16, 128, 128, 3)` — dữ liệu train
   - `train_y.npy` shape `(N,)` — nhãn (0 = NonViolence, 1 = Violence)
   - Tương tự cho `val` và `test`

**Frame Skipping (tối ưu hóa):**
Nếu video có nhiều frame hơn 16, hệ thống **lấy mẫu nhảy cóc đều** thay vì cắt bỏ cứng nhắc → bao quát ngữ cảnh hành động toàn clip.

### 4.4 Hàm mất mát — Focal Loss

Thay vì Binary Crossentropy thông thường, đề tài dùng **Focal Loss**:

$$FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| $\gamma = 2.0$ | Giảm trọng số mẫu dễ | Mô hình tập trung hơn vào các clip khó phân biệt (thể thao, võ thuật nhầm lẫn với bạo lực) |
| $\alpha = 0.50$ | Cân bằng lớp | Phòng ngừa thiên vị khi số mẫu Violence ≠ NonViolence |

> **Tại sao Focal Loss tốt hơn?** Dataset violence thường **mất cân bằng** (NonViolence >> Violence). BCE sẽ thiên về dự đoán an toàn liên tục → bỏ sót bạo lực. Focal Loss ép mô hình học các mẫu khó.

### 4.5 Chiến lược huấn luyện 2 pha (Two-Phase Training)

**Lý do chia 2 pha:**
Nếu mở khóa MobileNetV2 từ đầu và train cùng lúc → gradient từ LSTM ngẫu nhiên sẽ **phá hỏng** trọng số ImageNet đã được pre-train tốt.

```
╔══════════════════════════════════════════════════════╗
║  PHA 1 (Epochs 1-25): FROZEN BACKBONE               ║
║  ─────────────────────────────────────────────────  ║
║  • MobileNetV2 → Khóa hoàn toàn (trainable=False)  ║
║  • Chỉ train: BiLSTM + LSTM + Dense head            ║
║  • Learning Rate: 5e-4 (học nhanh phần mới)         ║
║  • Mục tiêu: LSTM học cách đọc chuỗi spatial features║
╚══════════════════════════════════════════════════════╝
              ↓ (sau pha 1 ổn định)
╔══════════════════════════════════════════════════════╗
║  PHA 2 (Epochs 1-15): FINE-TUNING                   ║
║  ─────────────────────────────────────────────────  ║
║  • Mở khóa 30 lớp cuối MobileNetV2                  ║
║  • Train toàn bộ mạng                               ║
║  • Learning Rate: 2e-5 (học rất chậm, tinh chỉnh)   ║
║  • Mục tiêu: Backbone thích nghi với dataset bạo lực║
╚══════════════════════════════════════════════════════╝
```

**Mixed Precision Training (`float16`):**
- Tính toán dùng `float16` → Tiết kiệm **50% VRAM** GPU.
- Trọng số vẫn lưu `float32` → Đảm bảo độ chính xác.
- Kết quả: Có thể train Batch Size = 16 thay vì 8.

### 4.6 Callbacks tự động hóa

| Callback | Chức năng |
|---|---|
| `ModelCheckpoint` | Tự lưu model tốt nhất (theo `val_auc` cao nhất) |
| `EarlyStopping` | Dừng sớm nếu `val_auc` không tăng sau 7 epoch → tránh overfitting |
| `ReduceLROnPlateau` | Giảm Learning Rate 50% nếu `val_loss` không giảm sau 4 epoch |
| `CSVLogger` | Ghi log từng epoch ra `training/train_log.csv` |
| `TensorBoard` | Trực quan hóa đường cong train/val (xem bằng `tensorboard --logdir logs/`) |

### 4.7 Cơ chế suy luận thời gian thực (Inference)

```
Video đầu vào
   │
   ↓ Chia thành cửa sổ trượt 3 giây
   │
   ├─ [00:00 - 00:03] → Trích 16 frame → MobileNetV2 + BiLSTM → P_vis = 0.12
   ├─ [00:03 - 00:06] → Trích 16 frame → MobileNetV2 + BiLSTM → P_vis = 0.87 ← 🔴
   ├─ [00:06 - 00:09] → Trích 16 frame → MobileNetV2 + BiLSTM → P_vis = 0.65
   └─ ...
         │
         ▼
   P_visual = max(0.12, 0.87, 0.65, ...) = 0.87
```

Điều này cho phép **định vị chính xác giây nào** có hành vi bạo lực → Dashboard tua video đến đúng giây đó.

### 4.8 Tự động nhận diện môi trường chạy

```python
IS_KAGGLE = os.path.exists("/kaggle")  # Kaggle Notebook
IS_COLAB  = os.path.exists("/content") # Google Colab

# → Tự động cấu hình đường dẫn data & output phù hợp
# Local PC: D:\SICT MediaTech\models\violence_train.h5
# Kaggle  : /kaggle/working/models/violence_train.h5
# Colab   : /content/drive/MyDrive/SICT_MediaTech/models/
```

---

## 5. 🤝 Phân chia công việc (gợi ý trình bày)

| Module | Nội dung chính |
|---|---|
| **Thành viên bạn** | Pipeline tiền xử lý video, Kiến trúc MobileNetV2 + BiLSTM, Focal Loss, Chiến lược 2 pha, Frame Skipping |
| **Thành viên khác** | NLP tiếng Việt (ViHSD), TF-IDF, Calibrated Classifier |
| **Thành viên khác** | OCR subtitle, watermark filter, dedup hashing |
| **Thành viên khác** | Giao diện Dashboard HTML, FastAPI server, Brand Safety Score |

---

## 6. 💬 Câu hỏi hội đồng thường gặp — Gợi ý trả lời

**Q: Tại sao chọn MobileNetV2 mà không dùng ResNet, VGG?**
> MobileNetV2 được thiết kế tối ưu cho ứng dụng thực tế với tài nguyên hạn chế — nhẹ hơn 10x so với VGG16, tốc độ inference nhanh hơn, vẫn đủ độ chính xác khi fine-tune với dữ liệu bạo lực chuyên biệt.

**Q: Tại sao lấy 16 frames mà không lấy nhiều hơn?**
> 16 frames là điểm cân bằng giữa thông tin thời gian và chi phí tính toán. Với video 3 giây ở 30fps = 90 frames raw, lấy 16 frames đều phân bố đã đủ bắt chuyển động bạo lực (thường kéo dài ≥ 0.5 giây). Lấy nhiều hơn tăng VRAM nhưng không cải thiện đáng kể độ chính xác.

**Q: Tại sao dùng cửa sổ trượt 3 giây?**
> Hành vi bạo lực thường xảy ra trong vài giây. Cửa sổ 3 giây đủ dài để bắt được 1 chuỗi hành động hoàn chỉnh, nhưng ngắn đủ để định vị chính xác giây vi phạm. Kết quả trả về là timeline từng phân đoạn 3s, không phải chỉ 1 con số cho cả video.

**Q: Mô hình được train trên dữ liệu gì?**
> Dataset violence_data gồm các video được gán nhãn thủ công chia thành 3 tập train/val/test, có cả lớp Violence và NonViolence, đảm bảo mô hình không chỉ nhận ra bạo lực mà còn không báo nhầm cảnh thể thao, hành động thông thường.

**Q: Tại sao cần Mixed Precision?**
> Mô hình có TimeDistributed(MobileNetV2) × 16 frames × batch = rất nặng. Mixed Precision giảm 50% bộ nhớ GPU, cho phép train batch lớn hơn → gradient ổn định hơn → hội tụ nhanh hơn.

---

## 7. 🚀 Chạy demo nhanh

```bash
# Khởi động server
cd "d:\SICT MediaTech\UI"
python -m uvicorn server:app --reload --port 8000

# Mở trình duyệt
http://127.0.0.1:8000
```

**Quy trình demo:**
1. Upload video mẫu (thư mục `video test/`) → `vd1.mp4` (bình thường), `vd3.mp4` (có bạo lực).
2. Nhập caption mẫu.
3. Bấm **"Chạy lại phân tích"**.
4. Hệ thống trả về điểm Brand Safety Score, timeline phân đoạn 3s, và phụ đề OCR nhận diện được.
5. Bấm vào từng ô thời gian → video tua đến đúng giây đó.
