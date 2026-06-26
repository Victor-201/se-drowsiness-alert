---
marp: true
theme: uncover
class: invert
---

# <!--fit--> Ứng Dụng Cảnh Báo Phát Hiện Buồn Ngủ Lái Xe

**Môn:** Xử Lý Ảnh & Thị Giác Máy Tính — 121036  
**ĐH Giao thông Vận tải TP.HCM**

---

## Vấn Đề & Động Lực

<!-- Mục 1: Đặt vấn đề -->

- ~20% tai nạn giao thông liên quan đến buồn ngủ (WHO)
- Phát hiện sớm dấu hiệu buồn ngủ → giảm tai nạn
- **Mục tiêu:** Hệ thống real-time phát hiện nhắm mắt, ngáp, nghiêng đầu

**Giả thuyết:** EAR=0.22, MAR=0.30, HEAD_TILT=15° cho F1 ≥ 0.85

---

## Pipeline Tổng Quan

```
Camera → Tiền xử lý → Face Detection → Landmarks → ROI → Phân loại → Cảnh báo
  (BGR)   (Gray+CLAHE)  (HOG+SVM)   (68 điểm)  (Mắt/Miệng) (EAR/MAR)  (Âm thanh)
```

**4 kỹ thuật từ chương trình học:**  
- Ch.2: Lọc ảnh, CLAHE  
- Ch.3: 68 facial landmarks  
- Ch.4: Phân đoạn ROI  
- Ch.5: Phân loại threshold

---

## Demo

![height:350px](https://via.placeholder.com/640x480/1a1a2e/ffffff?text=Demo+Live)

**Tính năng:**
- 👁️ EAR — phát hiện nhắm mắt
- 👄 MAR — phát hiện ngáp
- 🤕 Head Pose — phát hiện nghiêng đầu
- 🔊 Cảnh báo âm thanh + hình ảnh

---

## Kết Quả Khảo Sát EAR

| Ngưỡng | Precision | Recall | F1 |
|--------|-----------|--------|----|
| 0.16 | 1.000 | 0.467 | 0.636 |
| **0.22** | **0.944** | **0.900** | **0.921** |
| 0.26 | 0.667 | 0.967 | 0.789 |

<!-- 
Nhận xét: 
- EAR=0.22 tối ưu (F1=0.921)
- EAR thấp: bỏ sót nhiều
- EAR cao: dương tính giả
-->

---

## Kết Quả Khảo Sát MAR & Head Tilt

**MAR:**
| Ngưỡng | F1 |
|--------|----|
| 0.25 | 0.769 |
| **0.30** | **0.909** |
| 0.40 | 0.727 |

**Head Tilt:**
| Ngưỡng | F1 |
|--------|----|
| 10° | 0.706 |
| **15°** | **0.857** |
| 20° | 0.870 |

---

## Kết Quả Tổng Thể

<!-- Confusion Matrix summary -->

| Chỉ số | Giá trị |
|---------|---------|
| Accuracy | **0.950** |
| Precision | **0.921** |
| Recall | **0.933** |
| F1-score | **0.927** |
| FP Rate | 0.027 |

✅ **Đạt mục tiêu F1 ≥ 0.85**

---

## Đối Chiếu Giả Thuyết

| Giả thuyết | Kết quả |
|------------|---------|
| EAR thấp → giảm nhạy ✅ | Recall 0.967 → 0.467 |
| MAR=0.30 cho F1 tốt nhất ✅ | F1=0.909 |
| HEAD_TILT=15° cho F1 tốt ✅ | F1=0.857 |
| F1 tổng thể ≥ 0.85 ✅ | **F1=0.927** |

---

## Thảo Luận

**Hiệu quả:**
- Kết hợp đa chỉ số (EAR + MAR + head pose + tần suất)
- Dynamic threshold cho blink detection
- Calibration theo từng người

**Hạn chế:**
- Nhạy với ánh sáng yếu
- Khó phát hiện khi đeo kính
- Mất dấu khi góc quay > 45°

---

## Kết Luận

- ✅ Pipeline hoàn chỉnh với 4 kỹ thuật CV
- ✅ Tham số tối ưu: EAR=0.22, MAR=0.30, HEAD_TILT=15°
- ✅ F1-score **0.927** — vượt mục tiêu
- ✅ 1.200+ dòng Python, kiến trúc MVC rõ ràng
- 📊 Notebook + Báo cáo đầy đủ thí nghiệm

---

## Q&A

<!--fit--> Cảm ơn thầy và các bạn!

