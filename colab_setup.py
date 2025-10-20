"""
Google Colab用のセットアップスクリプト
MVTec-ADデータセットをGoogle Driveから読み込んで実行する
"""

import os
import zipfile
from google.colab import drive
import gdown

def setup_colab_environment():
    """Colab環境のセットアップ"""
    print("🚀 Colab環境をセットアップ中...")
    
    # Google Driveをマウント
    print("📁 Google Driveをマウント中...")
    drive.mount('/content/drive')
    
    # 必要なパッケージをインストール
    print("📦 パッケージをインストール中...")
    os.system("pip install -r requirements.txt")
    
    return True

def download_mvtec_from_drive(drive_path=None):
    """
    Google DriveからMVTec-ADデータセットをダウンロード
    
    Args:
        drive_path: Google Drive内のMVTec-ADのパス
                   例: '/content/drive/MyDrive/m2/data/mvtec'
    """
    print("📥 MVTec-ADデータセットをダウンロード中...")
    
    # dataディレクトリを作成
    os.makedirs('data', exist_ok=True)
    
    if drive_path and os.path.exists(drive_path):
        print(f"📂 Google Driveからデータを読み込み: {drive_path}")
        
        if drive_path.endswith('.zip'):
            # ZIP形式の場合は展開
            with zipfile.ZipFile(drive_path, 'r') as zip_ref:
                zip_ref.extractall('data/')
            print("✅ データセットの展開完了")
        else:
            # フォルダの場合はコピー（内容をmvtec_anomaly_detectionにコピー）
            os.system(f"cp -r '{drive_path}' data/mvtec_anomaly_detection")
            print("✅ データセットのコピー完了")
    else:
        print("⚠️  Google Driveパスが見つかりません")
        print("代替方法として公式サイトからダウンロードします...")
        
        # 公式サイトからダウンロード（時間がかかる可能性）
        url = "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/mvtec_anomaly_detection.tar.xz"
        print("📡 公式サイトからダウンロード中...")
        os.system(f"wget -O mvtec_anomaly_detection.tar.xz '{url}'")
        os.system("tar -xf mvtec_anomaly_detection.tar.xz -C data/")
        os.system("rm mvtec_anomaly_detection.tar.xz")
        print("✅ データセットのダウンロード・展開完了")
    
    # データセット構造を確認
    if os.path.exists('data/mvtec_anomaly_detection'):
        objects = os.listdir('data/mvtec_anomaly_detection')
        objects = [obj for obj in objects if os.path.isdir(f'data/mvtec_anomaly_detection/{obj}')]
        print(f"📊 検出されたオブジェクト数: {len(objects)}")
        print(f"オブジェクト一覧: {objects[:5]}..." if len(objects) > 5 else f"オブジェクト一覧: {objects}")
        return True
    else:
        print("❌ データセットの設定に失敗しました")
        return False

def run_anomaly_detection(shots=1, num_seeds=1, model_name="dinov2_vitb14"):
    """
    異常検知を実行
    
    Args:
        shots: ショット数（1-shot, 2-shot, etc. -1でfull-shot）
        num_seeds: シード数
        model_name: モデル名
    """
    print(f"🔍 異常検知を実行中...")
    print(f"  - ショット数: {shots}")
    print(f"  - シード数: {num_seeds}")
    print(f"  - モデル: {model_name}")
    
    cmd = f"""python run_anomalydino.py \
        --dataset MVTec \
        --data_root data/mvtec_anomaly_detection \
        --model_name {model_name} \
        --resolution 448 \
        --shots {shots} \
        --num_seeds {num_seeds} \
        --preprocess informed \
        --device cuda:0"""
    
    print(f"実行コマンド: {cmd}")
    os.system(cmd)
    print("✅ 実行完了")

if __name__ == "__main__":
    # 基本的な使用例
    print("=== AnomalyDINO Colab Setup ===")
    
    # 環境セットアップ
    setup_colab_environment()
    
    # データセットダウンロード
    # Google Driveのパスを指定（実際のパスに変更してください）
    drive_path = "/content/drive/MyDrive/m2/data/mvtec"  # 例
    download_mvtec_from_drive(drive_path)
    
    # 異常検知実行
    run_anomaly_detection(shots=1, num_seeds=1)