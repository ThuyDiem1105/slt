# SignMusketeers paper-like reproduction with pretrained DINOv2

Bộ code này dùng để thực nghiệm lại bài báo SignMusketeers gần nhất có thể khi không truy cập được GitHub/checkpoint chính thức.

## Ý tưởng

Pipeline mô phỏng bài báo:

```text
Video frame
→ MediaPipe Holistic
→ crop face / left hand / right hand
→ pretrained DINOv2-small lấy feature 384 chiều cho từng crop
→ normalize upper-body pose thành 14 chiều
→ project 4 stream: face, left hand, right hand, pose
→ concatenate
→ T5.1.1 sinh câu tiếng Anh
```

Bản cải tiến:

```text
Giống paper-like model
+ confidence-aware gating cho 4 stream
```

## Khác biệt với bài báo gốc

Bài báo gốc có bước self-supervised continue pretraining DINOv2-Face và DINOv2-Hand trên face/hand crops. Bộ code này dùng **pretrained DINOv2 `facebook/dinov2-small`** do không có checkpoint chính thức. Vì vậy, đây là bản gần nhất với setting ablation kiểu `Crop + original DINOv2`, không phải full reproduction 100%.

## Cấu trúc data của bạn

Bạn nói dữ liệu đã giải nén ra folder cùng tên zip, trong mỗi folder có `Raw_videos/`. Hãy đặt hoặc sửa đường dẫn trong `configs/default.yaml` như sau:

```text
data_raw/
├── train_rgb_front_clips/
│   └── Raw_videos/
├── val_rgb_front_clips/
│   └── Raw_videos/
├── test_rgb_front_clips/
│   └── Raw_videos/
├── how2sign_realigned_train.csv
├── how2sign_realigned_val.csv
└── how2sign_realigned_test.csv
```

Nếu tên folder khác, chỉ cần sửa:

```yaml
data:
  train_video_dir: .../Raw_videos
  val_video_dir: .../Raw_videos
  test_video_dir: .../Raw_videos
  train_csv: ...csv
  val_csv: ...csv
  test_csv: ...csv
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy 1% mỗi split

Linux / Colab / Modal:

```bash
bash run_1pct_experiment.sh
```

Windows:

```bat
run_1pct_experiment.bat
```

## Muốn tăng data

Chỉ sửa dòng này trong `configs/default.yaml`:

```yaml
prepare:
  fraction_per_split: 0.01
```

Ví dụ:

```yaml
fraction_per_split: 0.05   # 5% mỗi file train/val/test
fraction_per_split: 0.10   # 10%
fraction_per_split: 1.00   # toàn bộ
```

## Kết quả

Sau khi chạy xong, xem:

```text
outputs/pred_paper_like_pretrained_dinov2.csv
outputs/pred_proposed_confidence_aware.csv
```

Checkpoint:

```text
checkpoints/paper_like_pretrained_dinov2/best.pt
checkpoints/proposed_confidence_aware/best.pt
```

## Nên viết trong báo cáo

> Do không truy cập được mã nguồn/checkpoint chính thức, chúng tôi xây dựng một bản thực nghiệm lại SignMusketeers ở mức paper-like. Mô hình giữ các thành phần chính gồm face crop, hand crops, upper-body pose, pretrained DINOv2 và T5. Vì không có DINOv2-F/DINOv2-H đã self-supervised pretrain, chúng tôi sử dụng pretrained DINOv2 gốc và xem đây là biến thể gần với ablation `Crop + original DINOv2`. Trên nền đó, chúng tôi đề xuất confidence-aware gating để giảm ảnh hưởng của các stream kém tin cậy.
