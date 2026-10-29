# Interview notes – Detection-of-defective-railway-tracks


**One-line pitch:** an end-to-end system that classifies railway-track photos into five defect categories and
fairly compares MobileNetV2 vs EfficientNetB0 on a leakage-free split, served through FastAPI and a Next.js dashboard.

## Decisions you can defend
* **Single-label classification** because the data is one-folder-per-class with no boxes. Limitation: images with several defects. Next step: detection (YOLO) or multi-label.
* **Transfer learning in two phases** (frozen backbone -> fine-tune last blocks, small LR): few images, so reuse ImageNet features and avoid destroying them.
* **Leakage control:** MD5 + perceptual hash groups; groups never straddle train/val/test. Near-duplicate threshold kept strict (<=1 bit of 64) because track photos look alike – a looser threshold merged distinct photos during testing.
* **Model selection on validation macro-F1, final numbers on an untouched test set.** Macro-F1 (not accuracy) because classes may be imbalanced; class-weighted loss handles imbalance.
* **Winner rule is explicit and pre-declared:** higher macro-F1, unless within 0.01, then lower CPU latency.
* **Low-confidence handling:** threshold + top-2 margin. Softmax is not calibrated – say so.
* **Architecture:** PyTorch cannot run in a Vercel function (bundle limit 250 MB / 500 MB Python, 4.5 MB request body). The trained checkpoints are converted to ONNX and executed in the browser with ONNX Runtime Web, so the whole product is one static Vercel app. Preprocessing is a bit-exact port of Pillow's bilinear resize, verified against Pillow and against PyTorch predictions.
* **Honesty guarantees:** the UI never shows numbers that were not produced by `train.py` / `evaluate.py`.

## Bugs found by testing (good stories)
* Checkpoints saved with `torch.__version__` could not be loaded with `weights_only=True` (it is a `TorchVersion` object). Fixed by sanitising metadata; regression test added.
* Near-duplicate threshold of 4 bits discarded most of a synthetic dataset as "ambiguous"; reduced to 1.
* A mixed-unit efficiency chart (ms/MB/min on one axis) hid the training-time bars; split into three charts.

## Likely questions
* *Why macro-F1?* Equal weight per class; accuracy hides weak minority classes.
* *Why is test accuracy lower than validation?* Model selection overfits the validation set slightly; the test set is the unbiased estimate.
* *How would you improve it?* More data per class, Grad-CAM to check it looks at the defect, calibration, detection model.
* *Is it safe to deploy for real inspection?* No – decision support only; needs validation on real field data and human review.
