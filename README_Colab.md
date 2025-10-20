# AnomalyDINO Google Colab実行版

論文「AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2」をGoogle Colabで実行するためのリポジトリです。

## 🚀 クイックスタート

### 1. 事前準備
1. MVTec-ADデータセットをGoogle Driveにアップロード
   - 公式サイトからダウンロード: https://www.mvtec.com/company/research/datasets/mvtec-ad
   - Google Driveに`mvtec_anomaly_detection.zip`としてアップロード

2. Google Colab Pro/Pro+でA100を選択（推奨）

### 2. 実行手順
1. `AnomalyDINO_Colab.ipynb`をGoogle Colabで開く
2. ランタイム → ランタイムタイプを変更 → GPU（A100推奨）
3. ノートブックのセルを順番に実行

## 📊 期待される結果

**1-shot設定での結果:**
- Mean AUROC: **96.6% ± 数%**（論文値）
- 実行時間: A100で約30分（1シード）〜3時間（3シード）

## 🛠️ コマンドライン実行

ノートブックではなく直接実行したい場合:

```bash
# リポジトリクローン
git clone https://github.com/YOUR_USERNAME/AnomalyDINO-Colab.git
cd AnomalyDINO-Colab

# 環境セットアップ
pip install -r requirements.txt

# Google Driveマウント（Colab環境）
from google.colab import drive
drive.mount('/content/drive')

# データセットセットアップ
python -c "from colab_setup import *; download_mvtec_from_drive('/content/drive/MyDrive/mvtec_anomaly_detection.zip')"

# 異常検知実行
python run_anomalydino.py --dataset MVTec --data_root data/mvtec_anomaly_detection --model_name dinov2_vitb14 --shots 1 --num_seeds 1 --preprocess informed
```

## 📁 ファイル構成

```
AnomalyDINO-Colab/
├── src/                    # 核となるソースコード
├── run_anomalydino.py      # メイン実行スクリプト
├── colab_setup.py          # Colab用セットアップスクリプト
├── AnomalyDINO_Colab.ipynb # Colab実行用ノートブック
├── requirements.txt        # 依存関係
└── README_Colab.md        # このファイル
```

## ⚙️ パラメータ説明

- `--shots 1`: 1-shot設定（正常画像1枚使用）
- `--num_seeds 3`: 3つのシードで評価
- `--model_name dinov2_vitb14`: DINOv2-ViT-B/14モデル
- `--resolution 448`: 入力画像解像度
- `--preprocess informed`: オブジェクト特化前処理

## 🎯 MVTec-AD以外のデータセット

VisAデータセットでも実行可能:
```bash
python run_anomalydino.py --dataset VisA --data_root data/VisA_pytorch/1cls/
```

## 📝 引用

```bibtex
@inproceedings{damm2024anomalydino,
    title={AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2}, 
    author={Simon Damm and Mike Laszkiewicz and Johannes Lederer and Asja Fischer},
    booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV 2025)},
    year={2025},
    url={https://arxiv.org/abs/2405.14529}, 
}
```

## 🤝 サポート

問題が発生した場合は、Issue をご報告ください。