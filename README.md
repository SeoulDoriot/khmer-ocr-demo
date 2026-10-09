# Khmer OCR Architecture 3 demo

CNN + Transformer encoder + CTC, recognising cropped text lines.
Selected checkpoint: `best.pt`, step 87,000 from the completed 100,000-step run.
Saved macro validation character error rate: 8.23%. This validation set was used for checkpoint selection and is not an independent benchmark.

## Teacher demo

[Open in Google Colab](https://colab.research.google.com/github/SeoulDoriot/khmer-ocr-demo/blob/main/Demo.ipynb), then click **Runtime → Run all**. The notebook downloads the checkpoint and selected examples and computes predictions on CPU.

- [Model files](https://huggingface.co/Seoul12/khmer-ocr-architecture3-best)
- [Selected demo dataset](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo)

## Run locally

Install Python 3.10 or newer, then:

```sh
pip install -r requirements.txt
python inference.py --checkpoint best.pt --image your_line_image.png
python inference.py --checkpoint best.pt --manifest demo/manifest.json
```

The demo dataset contains selected successful examples. Its exact-match result is not a claim of 100% accuracy on unseen images. Predictions are computed from the checkpoint at runtime. Full screening results accompany the examples.

Use cropped single text lines. Full pages require detection and line segmentation. Handwriting and scene photos are less reliable than printed lines.

Source checkpoint: https://www.kaggle.com/code/seoulvy/khmer-ocr-architecture-3-100k-continuation/output?scriptVersionId=356010649

Model download destination: https://huggingface.co/Seoul12/khmer-ocr-architecture3-best
Public repositories; no Hugging Face account is required to download.
