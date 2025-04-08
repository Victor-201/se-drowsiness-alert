Dưới đây là nội dung file `README.md` được viết dựa trên các tài liệu bạn cung cấp. Nội dung được trình bày rõ ràng, chuyên nghiệp và phù hợp để đưa lên GitHub. Tôi đã tối ưu hóa để người dùng dễ hiểu và có thể bắt đầu sử dụng dự án ngay lập tức.

---

# 🚗 Driver Drowsiness Alert App

Ứng dụng cảnh báo tài xế ngủ gật sử dụng xử lý ảnh và trí tuệ nhân tạo để phát hiện dấu hiệu buồn ngủ hoặc mất tập trung khi lái xe, nhằm giảm thiểu nguy cơ tai nạn giao thông.

## 📝 Giới thiệu
Ứng dụng này giám sát khuôn mặt tài xế trong thời gian thực, phát hiện các dấu hiệu như nhắm mắt lâu hoặc gục đầu, và phát âm thanh cảnh báo khi cần thiết. Dự án được thiết kế để dễ dàng triển khai trên các thiết bị Android.

## 🎯 Tính năng chính
- 👁️ Nhận diện khuôn mặt và mắt của tài xế.
- ⏳ Theo dõi thời gian nhắm mắt để phát hiện trạng thái buồn ngủ.
- 🔔 Phát âm thanh cảnh báo khi tài xế có dấu hiệu ngủ gật.
- 📊 Ghi lại dữ liệu hành vi (tùy chọn mở rộng).

## 🔧 Công nghệ sử dụng
| Công nghệ        | Mục đích sử dụng                          |
|------------------|-------------------------------------------|
| Python           | Ngôn ngữ lập trình chính                  |
| OpenCV           | Xử lý ảnh, nhận diện khuôn mặt và mắt     |
| TensorFlow/Keras | Mô hình AI phát hiện buồn ngủ (CNN)       |
| Plyer            | Tạo thông báo và âm thanh cảnh báo        |
| Kivy/KivyMD      | Thiết kế giao diện ứng dụng Android       |
| Buildozer        | Đóng gói ứng dụng thành file APK          |

## 📋 Yêu cầu cài đặt
- Python 3.8+
- Buildozer (cho Android, chỉ chạy trên Linux)
- Hệ điều hành: Linux (để build APK), Windows/Mac (để phát triển)
- Git

## 📲 Hướng dẫn cài đặt và chạy
### 1. Tải mã nguồn
```bash
git clone https://github.com/Victor-201/se-drowsiness-alert.git
cd se-drowsiness-alert
```

### 2. Cài đặt môi trường Python
Tạo môi trường ảo (khuyến nghị) và cài đặt các thư viện:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Chạy ứng dụng trên máy tính
```bash
python main.py
```
- Nhấn "Start" để bắt đầu giám sát.
- Nhấn "Stop" để dừng.

### 4. Build APK cho Android (Linux)
```bash
buildozer init
buildozer -v android debug
```
File APK sẽ được tạo tại: `bin/se-drowsiness-alert-debug.apk`.

## ⚙️ Cấu hình
- File cấu hình: `src/configs/config.py`
- Điều chỉnh các thông số như `EAR_THRESHOLD` (ngưỡng EAR), `ALERT_SOUND_FILE` (đường dẫn âm thanh cảnh báo), v.v.

## 📂 Cấu trúc thư mục
```
driver-drowsiness-alert-app/
├── assets/              # File âm thanh cảnh báo
├── bin/                 # File APK sau khi build
├── data/                # Mô hình và file calibration
├── src/                 # Mã nguồn chính
│   ├── configs/         # Cấu hình
│   ├── core/            # Logic xử lý
│   └── ui/              # Giao diện ứng dụng
├── main.py              # File chạy chính
├── requirements.txt     # Danh sách thư viện
└── README.md            # Tài liệu này
```

## 💡 Góp ý và đóng góp
- Mở issue để báo lỗi hoặc đề xuất tính năng mới.
- Fork dự án, tạo pull request để đóng góp mã nguồn.
