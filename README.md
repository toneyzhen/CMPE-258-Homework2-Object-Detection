# CMPE-258-Homework2-Object-Detection

Homework 2 for CMPE 258 (Deep Learning) at San Jose State University, Option 1: Training Optimization. The task is to take a 2D object detector, change how it is trained, and compare the result against a baseline using COCO-style metrics.

Both notebooks use [Ultralytics YOLOv8](https://docs.ultralytics.com/) and run in Google Colab.

## Notebooks

| Notebook | Dataset | Baseline | Improved |
|---|---|---|---|
| [`cppe5_yolov8_training_1.ipynb`](cppe5_yolov8_training_1.ipynb) | [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5): medical protective equipment, 5 classes | YOLOv8n, 50 epochs, default settings | YOLOv8n, up to 150 epochs, cosine learning rate, mosaic off for the last 15 epochs, early stopping, plus extra images from a second dataset |
| [`YOLOv8_COCO_Training_fixed.ipynb`](YOLOv8_COCO_Training_fixed.ipynb) | [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/): 128 images from COCO, 108 to train and 20 to validate | YOLOv8n with its pretrained COCO weights, no fine-tuning | YOLOv8s fine-tuned for 200 epochs with AdamW, cosine learning rate, MixUp, a longer warmup, and the first 10 layers frozen |

The COCO128 notebook also sets up a Label Studio annotation workflow for the 20 validation images.

## Results

These are the numbers saved in the notebooks' outputs.

**CPPE-5** (validation images held out from the CPPE-5 training set)

| Metric | Baseline | Improved |
|---|---|---|
| mAP@50 | 0.8595 | 0.8659 |
| mAP@50:95 | 0.5842 | 0.5941 |
| Precision | 0.8767 | 0.8824 |
| Recall | 0.7927 | 0.8070 |

The improved run is slightly better overall. Per class, Goggles gained the most (0.8074 to 0.8419 AP@50) and Mask dropped slightly (0.8950 to 0.8847).

**COCO128** (20 validation images)

| Metric | Baseline | Improved |
|---|---|---|
| mAP@0.5 | 0.6190 | 0.4354 |
| mAP@0.5:0.95 | 0.4379 | 0.2550 |
| Precision | 0.5773 | 0.4963 |
| Recall | 0.5716 | 0.3213 |

Here the fine-tuned model did worse than the baseline. The baseline's weights were pretrained on the full COCO dataset, and fine-tuning a larger model on 108 images did not match them.

## Caveats in the saved runs

- **CPPE-5:** the notebook describes merging about 2,000 images from `keremberke/protective-equipment-detection`, but the saved output of that cell reports 0 images merged. The improved result above therefore reflects the training-schedule changes on CPPE-5 alone.
- **CPPE-5:** the saved output of the split cell shows 800 training and 400 validation images, not the intended 800 and 200, which suggests the cell was run twice in that session.
- **COCO128:** the validation labels are the original COCO labels (`USE_ORIGINAL_AS_FALLBACK = True`), not labels drawn by hand in Label Studio.

## Running the notebooks

1. Open a notebook with its **Open in Colab** badge.
2. Set the runtime to a **T4 GPU** (Runtime → Change runtime type).
3. Run the cells in order. The datasets download inside the notebook. The COCO128 notebook asks you to restart the session once after its environment-fix cell.
