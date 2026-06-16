# CA-CSA cho Dịch Ngôn Ngữ Ký Hiệu

Repository này chứa code thực nghiệm cho bài toán **Sign Language Translation (SLT)**, lấy cảm hứng từ bài báo:

> **SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale**

Mục tiêu chính của project là thực nghiệm và đánh giá module cải tiến:

> **CA-CSA: Confidence-Aware Cross-Stream Attention**

CA-CSA cải tiến bước hợp nhất đa luồng bằng cách cho các stream mặt, tay trái, tay phải và pose tương tác với nhau, đồng thời xét đến độ tin cậy của từng stream.

---

## 1. Tổng quan project

Bài toán:

```text
Input  : video ngôn ngữ ký hiệu ASL
Output : câu tiếng Anh tương ứng
```

Pipeline thực nghiệm gồm 4 stream chính:

```text
Face stream
Left hand stream
Right hand stream
Upper-body pose stream
```

Baseline paper-like dùng cách fusion đơn giản:

```text
concat + linear projection
```

Mô hình đề xuất CA-CSA thay thế bước đó bằng:

```text
Confidence-Aware Cross-Stream Attention
```

Mục tiêu là so sánh:

```text
Paper-like baseline fusion
vs
CA-CSA fusion
```

trong cùng một điều kiện dữ liệu, feature extractor và decoder.

---

## 2. Lưu ý quan trọng

Đây **không phải official reproduction đầy đủ** của SignMusketeers.

Do không truy cập được mã nguồn/checkpoint chính thức của bài báo, project này xây dựng một baseline mô phỏng ở mức nhẹ hơn:

- Dùng pretrained `facebook/dinov2-small`
- Không self-supervised pretrain lại DINOv2-Face / DINOv2-Hand
- Dùng `t5-small` thay vì T5.1.1-Base
- Chạy trên 50% dữ liệu How2Sign
- Mục tiêu chính là kiểm tra tác động của CA-CSA trong điều kiện kiểm soát

Vì vậy, thực nghiệm này nên được hiểu là:

```text
controlled ablation / lightweight paper-like experiment
```

Không nên hiểu là reproduce đầy đủ kết quả của SignMusketeers gốc.

---

## 3. Hai mô hình được so sánh

### 3.1 Paper-like baseline

Run name:

```text
paper_like_pretrained_dinov2
```

Kiến trúc:

```text
face feature       -> projection
left hand feature  -> projection
right hand feature -> projection
pose14             -> projection
        ↓
concatenate
        ↓
linear projection
        ↓
T5-small
        ↓
English sentence
```

Model này mô phỏng hướng fusion của SignMusketeers ở mức lightweight.

### 3.2 CA-CSA full

Run name:

```text
ca_csa_full
```

Kiến trúc:

```text
face / left hand / right hand / pose
        ↓
stream-specific projection
        ↓
stream identity embedding
        ↓
learned confidence
        ↓
temporal confidence smoothing
        ↓
confidence-biased cross-stream attention
        ↓
confidence-weighted pooling
        ↓
T5-small
        ↓
English sentence
```

CA-CSA bổ sung các thành phần:

- Giữ định danh riêng cho từng stream
- Cho các stream tương tác bằng attention
- Học độ tin cậy của từng stream
- Dùng confidence làm bias trong attention
- Pooling theo độ tin cậy của stream

---

## 4. Cấu trúc thư mục

```text
.
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── 00_check_data.py
│   ├── 01_prepare_manifest.py
│   ├── 02_extract_signmusketeers_dinov2_features.py
│   ├── 03_train_t5.py
│   └── 05_collect_results.py
│
├── src/
│   └── smexp/
│       ├── config.py
│       ├── data.py
│       ├── manifest.py
│       ├── metrics.py
│       ├── model.py
│       └── video.py
│
├── requirements.txt
├── run_1pct_experiment.bat
├── run_1pct_experiment.sh
├── .gitignore
└── README.md
```

Các thư mục dữ liệu lớn như `data_raw/`, `features/`, `outputs/`, `checkpoints/` không được upload lên GitHub.

---

## 5. Cấu trúc dữ liệu yêu cầu

Dữ liệu cần được đặt cục bộ theo cấu trúc:

```text
data_raw/
├── train_rgb_front_clips/
│   └── Raw_videos/
│       ├── video_1.mp4
│       └── ...
│
├── val_rgb_front_clips/
│   └── Raw_videos/
│       ├── video_1.mp4
│       └── ...
│
├── test_rgb_front_clips/
│   └── Raw_videos/
│       ├── video_1.mp4
│       └── ...
│
├── how2sign_realigned_train.csv
├── how2sign_realigned_val.csv
└── how2sign_realigned_test.csv
```

Ví dụ tên video:

```text
_fZbAxSSbX4_0-5-rgb_front.mp4
_fZbAxSSbX4_1-5-rgb_front.mp4
```

Các file CSV cần có thông tin mapping giữa video clip và câu tiếng Anh, thường gồm các cột:

```text
VIDEO_ID
VIDEO_NAME
SENTENCE_ID
SENTENCE_NAME
SENTENCE
```

---

## 6. Cài đặt môi trường

Khuyến nghị dùng Python 3.10 hoặc 3.11.

Tạo virtual environment:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Nếu thiếu `torchvision`, cài thêm:

```bash
pip install torch torchvision torchaudio
```

---

## 7. Chạy toàn bộ thực nghiệm 50%

Trên Windows, chạy:

```bash
run_1pct_experiment.bat
```

File này thực hiện các bước:

```text
1. Kiểm tra đường dẫn dữ liệu
2. Tạo manifest 50%
3. Trích đặc trưng DINOv2 cho face/hand crops và pose14
4. Train paper-like baseline
5. Train CA-CSA full
6. Evaluate train / validation / test và lưu kết quả
```

---

## 8. Chạy từng bước thủ công

### Bước 1: Kiểm tra dữ liệu

```bash
python scripts\00_check_data.py --config configs\default.yaml
```

### Bước 2: Tạo manifest 1%

```bash
python scripts\01_prepare_manifest.py --config configs\default.yaml
```

### Bước 3: Trích đặc trưng DINOv2

```bash
python scripts\02_extract_signmusketeers_dinov2_features.py --config configs\default.yaml
```

### Bước 4: Train baseline

```bash
python scripts\03_train_t5.py --config configs\default.yaml --run_name paper_like_pretrained_dinov2 --mode paper
```

### Bước 5: Train CA-CSA

```bash
python scripts\03_train_t5.py --config configs\default.yaml --run_name ca_csa_full --mode ca_csa_full
```

### Bước 6: Gom kết quả train / val / test

```bash
python scripts\05_collect_results.py --config configs\default.yaml --runs paper_like_pretrained_dinov2 ca_csa_full
```

---

## 9. Kết quả đầu ra

Sau khi chạy xong, kết quả được lưu trong thư mục:

```text
outputs/
```

Các file chính:

```text
outputs/summary_all_splits.csv
outputs/pred_paper_like_pretrained_dinov2_train.csv
outputs/pred_paper_like_pretrained_dinov2_val.csv
outputs/pred_paper_like_pretrained_dinov2_test.csv
outputs/pred_ca_csa_full_train.csv
outputs/pred_ca_csa_full_val.csv
outputs/pred_ca_csa_full_test.csv
```

File `summary_all_splits.csv` chứa kết quả tổng hợp trên train, validation và test.

---

## 10. Thay đổi tỷ lệ dữ liệu

Mặc định project chạy trên 50% dữ liệu mỗi split.

Muốn tăng dữ liệu, mở:

```text
configs/default.yaml
```

Sửa:

```yaml
prepare:
  fraction_per_split: 0.5
```

Ví dụ:

```yaml
fraction_per_split: 0.05   # chạy 5%
fraction_per_split: 0.10   # chạy 10%
fraction_per_split: 1.00   # chạy toàn bộ dữ liệu
```

Sau khi đổi tỷ lệ dữ liệu, cần chạy lại từ bước tạo manifest và extract feature.

---

## 11. Setting thực nghiệm hiện tại

Thiết lập mặc định:

```text
Dataset            : How2Sign
Subset             : 50% mỗi split
Visual backbone    : pretrained DINOv2-small
Decoder            : T5-small
Streams            : face, left hand, right hand, pose14
Metrics            : BLEU, ROUGE-L
```

Hai mô hình được so sánh:

```text
paper_like_pretrained_dinov2
ca_csa_full
```

---

## 12. Ý nghĩa thực nghiệm

Thực nghiệm này được thiết kế để kiểm tra câu hỏi:

```text
Nếu giữ nguyên feature extractor và decoder,
việc thay concat + linear fusion bằng CA-CSA có giúp cải thiện chất lượng dịch không?
```

Do đó, điểm quan trọng không phải là tái hiện chính xác toàn bộ SignMusketeers, mà là so sánh công bằng giữa:

```text
same features + concat fusion
vs
same features + CA-CSA fusion
```

---

## 13. Lưu ý về GitHub

Ngoài các mục trên github còn các mục khác:

```text
data_raw/
data/
features/
outputs/
checkpoints/
.venv/
```

---

## 14. Hướng phát triển tiếp theo

Các bước tiếp theo để làm thực nghiệm đáng tin hơn:

- Tăng dữ liệu từ 50% lên 100% How2Sign
- Chạy nhiều random seeds
- Thêm ablation:
  - Cross-stream attention only
  - MediaPipe confidence bias
  - Learned confidence
  - Temporal smoothing
  - Full CA-CSA
- Thêm robustness test khi che tay, che mặt hoặc làm nhiễu pose
- Visualize attention và confidence theo thời gian
- Thử T5.1.1-Base nếu có đủ GPU

---

## 15. Citation

Base paper:

```text
SignMusketeers: An Efficient Multi-Stream Approach
for Sign Language Translation at Scale
Findings of ACL 2025
```

This project is a lightweight experimental extension focusing on CA-CSA fusion.
