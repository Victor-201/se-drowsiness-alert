# Ứng Dụng Cảnh Báo Phát Hiện Buồn Ngủ Lái Xe

Đồ án môn **Xử Lý Ảnh và Thị Giác Máy Tính** — Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh.

## Giới thiệu

Ứng dụng giám sát khuôn mặt tài xế qua webcam theo thời gian thực, phát hiện các dấu hiệu buồn ngủ (nhắm mắt lâu, ngáp, nghiêng đầu) và phát cảnh báo. Pipeline xử lý ảnh kết hợp các kỹ thuật từ Chương 2–5 của chương trình học.

## Kỹ thuật CV sử dụng

| Chương | Kỹ thuật | Chi tiết |
|--------|----------|----------|
| Ch. 2 | Xử lý ảnh | Chuyển grayscale, CLAHE cân bằng histogram thích ứng |
| Ch. 3 | Phát hiện đặc trưng | Haar Cascade (Viola-Jones) phát hiện khuôn mặt, dlib shape predictor 68 landmarks, Canny edge detection trên ROI mắt |
| Ch. 4 | Phân đoạn ảnh | Trích xuất ROI mắt và miệng dựa trên landmark, phân đoạn ngưỡng |
| Ch. 5 | Nhận dạng | Phân loại threshold-based: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), head pose |

## Yêu cầu cài đặt

- Python 3.8–3.11
- Webcam

## Hướng dẫn cài đặt và chạy

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python src/main.py
```

Tài nguyên model `shape_predictor_68_face_landmarks.dat` (~99MB) sẽ được tự động tải khi chạy lần đầu nếu chưa có trong thư mục `data/`.

## Usage

### Các chế độ chạy

```bash
# Giao diện Desktop Kivy (mặc định — đầy đủ chức năng)
python src/main.py
hoặc
python run.py

# Giao diện OpenCV (chế độ xem pipeline)
python src/main.py --opencv

# Bật lưu ảnh pipeline (chỉ với --opencv)
python src/main.py --opencv --save-pipeline

# Hiệu chỉnh ngưỡng EAR
python src/main.py --calibrate
```

### Phím tắt (OpenCV mode)

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát |
| `p` | Bật/tắt lưu pipeline stages (ảnh trung gian) |

### Jupyter notebook

Mở notebook thí nghiệm và khảo sát tham số:

```bash
jupyter notebook notebooks/experiments.ipynb
```

## Cấu trúc thư mục

```
se-drowsiness-alert/
├── assets/                   # Âm thanh, hình ảnh, font chữ
│   ├── fonts/
│   ├── images/
│   └── sounds/
├── data/                     # Model dlib, file hiệu chỉnh, cài đặt
├── notebooks/                # Jupyter notebook thí nghiệm + ảnh test
│   ├── experiments.ipynb
│   └── generate_test_face.py
├── src/
│   ├── __init__.py
│   ├── main.py               # Điểm vào ứng dụng (Kivy/OpenCV/Calibrate)
│   ├── configs/              # Cấu hình hệ thống
│   │   ├── config.py         #   Config — tham số mặc định
│   │   └── settings.py       #   Settings — cài đặt người dùng
│   ├── core/                 # Logic phát hiện và xử lý
│   │   ├── detector.py       #   Pipeline chính + Haar Cascade face detection
│   │   ├── facial_analyzer.py#   EAR, MAR, CLAHE, Canny, head pose
│   │   ├── alert_system.py   #   Cảnh báo overlay + âm thanh
│   │   └── model_manager.py  #   Quản lý model dlib (tự động tải)
│   ├── evaluation/           # Đánh giá định lượng
│   │   └── metrics.py
│   ├── exceptions/           # Exception classes
│   │   └── app_exceptions.py
│   └── ui/                   # Giao diện Desktop Kivy
│       ├── app.py            #   App chính (Kivy)
│       ├── widgets.py        #   Widget dùng chung (IconButton)
│       ├── styles.py         #   KV style strings
│       └── screens/
│           ├── main_screen.py#   Màn hình chính
│           └── settings_screen.py # Màn hình cài đặt
├── pipeline_output/          # Ảnh pipeline trung gian (khi bật --save-pipeline)
├── requirements.txt
└── README.md
```

## Cấu hình

Các tham số chính trong `src/configs/config.py`:
- `EAR_THRESHOLD`: Ngưỡng phát hiện mắt nhắm (mặc định 0.22)
- `EAR_CONSEC_FRAMES`: Số frame liên tiếp mắt nhắm để báo động (15)
- `YAWN_THRESHOLD`: Ngưỡng phát hiện ngáp (0.30)
- `YAWN_CONSEC_FRAMES`: Số frame liên tiếp ngáp để xác nhận (5)
- `HEAD_TILT_THRESHOLD`: Ngưỡng góc nghiêng đầu (15 độ)
- `HEAD_TILT_FRAMES`: Số frame liên tiếp nghiêng đầu để báo động (20)
- `BLINK_PER_MINUTE_THRESHOLD`: Tần suất nháy mắt/phút để cảnh báo mệt mỏi (25)
- `YAWN_PER_MINUTE_THRESHOLD`: Tần suất ngáp/phút để cảnh báo mệt mỏi (3)
- `NO_FACE_ALERT_FRAMES`: Số frame không thấy mặt để báo mất tập trung (20)
- `CALIBRATION_DURATION`: Thời gian hiệu chỉnh (giây, mặc định 5)
