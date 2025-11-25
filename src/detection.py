import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch

from src.utils import augment_image, dists2map, plot_ref_images
from src.post_eval import mean_top1p

def two_stage_detection(patch_score, stats_score, object_name, stage1_threshold=0.16, stage2_threshold=0.12):
    """
    2段階異常検知（シンプル版）
    
    Args:
        patch_score: パッチベースのTop1%スコア
        stats_score: 統計量ベースのスコア
        object_name: オブジェクト名
        stage1_threshold: Stage1閾値（パッチベース）
        stage2_threshold: Stage2閾値（統計量ベース）
    
    Returns:
        is_anomaly: 異常判定結果 (bool)
        final_score: 最終スコア (float)
        detection_method: 検知方法 (str)
    """
    
    # Stage 1: パッチベース検知
    if patch_score > stage1_threshold:
        return True, patch_score, "patch_based"
    
    # Stage 2: 統計量による補完検知
    elif stats_score > stage2_threshold:
        return True, stats_score, "statistics_based"
    
    # 正常判定
    else:
        return False, max(patch_score, stats_score), "normal"

def run_anomaly_detection(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        save_examples = False,
        masking = None,
        mask_ref_images = False,
        rotation = False,
        knn_metric = 'L2_normalized',
        knn_neighbors = 1,
        faiss_on_cpu = False,
        seed = 0,
        save_patch_dists = True,
        save_tiffs = False,
        stage1_threshold = 0.16,
        stage2_threshold = 0.12):
    """
    Main function to evaluate the anomaly detection performance of a given object/product.

    Parameters:
    - model: The backbone model for feature extraction (and, in case of DINOv2, masking).
    - object_name: The name of the object/product to evaluate.
    - data_root: The root directory of the dataset.
    - n_ref_samples: The number of reference samples to use for evaluation (k-shot). Set to -1 for full-shot setting.
    - object_anomalies: The anomaly types for each object/product.
    - plots_dir: The directory to save the example plots.
    - save_examples: Whether to save example images and plots. Default is True.
    - masking: Whether to apply DINOv2 to estimate the foreground mask (and discard background patches).
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - knn_neighbors: The number of nearest neighbors to consider. Default is 1.
    - seed: The seed value for deterministic sampling in few-shot setting. Default is 0.
    - save_patch_dists: Whether to save the patch distances. Default is True. Required to eval detection.
    - save_tiffs: Whether to save the anomaly maps as TIFF files. Default is False. Required to eval segmentation.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    
    # add 'good' to the anomaly types
    type_anomalies = object_anomalies[object_name]
    type_anomalies.append('good')

    # ensure that each type is only evaluated once
    type_anomalies = list(set(type_anomalies))

    # Extract reference features
    features_ref = []
    images_ref = []
    masks_ref = []
    vis_backgroud = []
    
    # Extract reference statistics for 2-stage detection
    stats_ref_mean = []  # 平均ベクトル集合
    stats_ref_std = []   # 標準偏差ベクトル集合

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if n_ref_samples == -1:
        # full-shot setting
        img_ref_samples = sorted(os.listdir(img_ref_folder))
    else:
        # few-shot setting, pick samples in deterministic fashion according to seed
        img_ref_samples = sorted(os.listdir(img_ref_folder))[seed*n_ref_samples:(seed + 1)*n_ref_samples]

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
    
    with torch.inference_mode():
        # start measuring time (feature extraction/memory bank set up)
        start_time = time.time()
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
            # load reference image...
            img_ref = f"{img_ref_folder}{img_ref_n}"
            image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # augment reference image (if applicable)...
            if rotation:
                img_augmented = augment_image(image_ref)
            else:
                img_augmented = [image_ref]
            for i in range(len(img_augmented)):
                image_ref = img_augmented[i]
                image_ref_tensor, grid_size1 = model.prepare_image(image_ref)
                features_ref_i = model.extract_features(image_ref_tensor)
                
                # compute background mask and discard background patches
                mask_ref = model.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=(mask_ref_images and masking))
                masked_features = features_ref_i[mask_ref]
                features_ref.append(masked_features)
                
                # Compute statistics for 2-stage detection
                if len(masked_features) > 0:  # マスクされたパッチが存在する場合のみ
                    # 統計量計算
                    patch_mean = masked_features.mean(axis=0)  # 全体平均特徴
                    patch_std = masked_features.std(axis=0)    # 全体ばらつき特徴
                    
                    # L2正規化
                    patch_mean = patch_mean / (np.linalg.norm(patch_mean) + 1e-8)
                    patch_std = patch_std / (np.linalg.norm(patch_std) + 1e-8)
                    
                    stats_ref_mean.append(patch_mean)
                    stats_ref_std.append(patch_std)
                
                if save_examples:
                    images_ref.append(image_ref)
                    vis_image_background = model.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                    masks_ref.append(mask_ref)
                    vis_backgroud.append(vis_image_background)
        
        features_ref = np.concatenate(features_ref, axis=0).astype('float32')
        
        # Convert statistics to numpy arrays
        if len(stats_ref_mean) > 0:
            stats_ref_mean = np.array(stats_ref_mean).astype('float32')
            stats_ref_std = np.array(stats_ref_std).astype('float32')
        else:
            # フォールバック：統計量が空の場合はダミーデータ
            stats_ref_mean = np.zeros((1, features_ref.shape[1]), dtype='float32')
            stats_ref_std = np.zeros((1, features_ref.shape[1]), dtype='float32')

        if faiss_on_cpu:
            # similariy search on CPU
            knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            # similariy search on GPU
            res = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
            # knn_index = faiss.IndexFlatL2(features_ref.shape[1])
            # knn_index = faiss.index_cpu_to_gpu(res, int(model.device[-1]), knn_index)


        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
        knn_index.add(features_ref)
        
        # Build statistics indices for 2-stage detection
        if faiss_on_cpu:
            stats_index_mean = faiss.IndexFlatL2(stats_ref_mean.shape[1])
            stats_index_std = faiss.IndexFlatL2(stats_ref_std.shape[1])
        else:
            stats_index_mean = faiss.GpuIndexFlatL2(res, stats_ref_mean.shape[1])
            stats_index_std = faiss.GpuIndexFlatL2(res, stats_ref_std.shape[1])
        
        # Normalize and add statistics to indices
        faiss.normalize_L2(stats_ref_mean)
        faiss.normalize_L2(stats_ref_std)
        stats_index_mean.add(stats_ref_mean)
        stats_index_std.add(stats_ref_std)

        # end measuring time (for memory bank set up; in seconds, same for all test samples of this object)
        time_memorybank = time.time() - start_time

        # plot some reference samples for inspection
        if save_examples:
            plots_dir_ = f"{plots_dir}/{object_name}/"
            plot_ref_images(images_ref, masks_ref, vis_backgroud, grid_size1, plots_dir_, title = "Reference Images", img_names = img_ref_samples)   
        
        inference_times = {}
        anomaly_scores = {}
        detailed_scores = {}  # Stage別の詳細スコアを記録

        idx = 0
        # Evaluate anomalies for each anomaly type (and "good")
        for type_anomaly in tqdm(type_anomalies, desc = f"processing test samples ({object_name})"):
            data_dir = f"{data_root}/{object_name}/test/{type_anomaly}"
            
            if save_patch_dists or save_tiffs:
                os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}", exist_ok=True)
            
            for idx, img_test_nr in enumerate(sorted(os.listdir(data_dir))):
                # start measuring time (inference)
                start_time = time.time()
                image_test_path = f"{data_dir}/{img_test_nr}"

                # Extract test features
                image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image_tensor2, grid_size2 = model.prepare_image(image_test)
                features2 = model.extract_features(image_tensor2)

                # Compute background mask
                if masking:
                    mask2 = model.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking)
                else:
                    mask2 = np.ones(features2.shape[0], dtype=bool)
                if save_examples and idx < 3:
                    vis_image_test_background = model.get_embedding_visualization(features2, grid_size2, mask2)

                # Discard irrelevant features
                features2 = features2[mask2]

                # Compute distances to nearest neighbors in M
                if knn_metric == "L2":
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = np.sqrt(distances)

                elif knn_metric == "L2_normalized":
                    faiss.normalize_L2(features2) 
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = distances / 2   # equivalent to cosine distance (1 - cosine similarity)

                output_distances = np.zeros_like(mask2, dtype=float)
                output_distances[mask2] = distances.squeeze()
                d_masked = output_distances.reshape(grid_size2)
                
                # Calculate patch-based anomaly score (existing method)
                patch_score = mean_top1p(output_distances.flatten())
                
                # Calculate statistics-based anomaly score (new method)
                if len(features2) > 0:
                    # テスト画像の統計量計算
                    test_mean = features2.mean(axis=0)
                    test_std = features2.std(axis=0)
                    
                    # L2正規化
                    test_mean = test_mean / (np.linalg.norm(test_mean) + 1e-8)
                    test_std = test_std / (np.linalg.norm(test_std) + 1e-8)
                    
                    # 統計量距離計算
                    test_mean_norm = test_mean.reshape(1, -1).astype('float32')
                    test_std_norm = test_std.reshape(1, -1).astype('float32')
                    
                    faiss.normalize_L2(test_mean_norm)
                    faiss.normalize_L2(test_std_norm)
                    
                    dist_mean, _ = stats_index_mean.search(test_mean_norm, k=1)
                    dist_std, _ = stats_index_std.search(test_std_norm, k=1)
                    
                    # 統計量スコア計算（コサイン距離変換）
                    stats_score = 0.7 * (dist_mean[0][0] / 2) + 0.3 * (dist_std[0][0] / 2)
                else:
                    stats_score = 0.0
                
                # 2段階異常検知
                is_anomaly, final_score, detection_method = two_stage_detection(
                    patch_score, stats_score, object_name, stage1_threshold, stage2_threshold
                )
                
                # save inference time
                torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
                inf_time = time.time() - start_time
                inference_times[f"{type_anomaly}/{img_test_nr}"] = inf_time
                anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = final_score

                # Stage別の詳細スコアを記録
                sample_key = f"{type_anomaly}/{img_test_nr}"
                detailed_scores[sample_key] = {
                    "patch_score": float(patch_score),
                    "stats_score": float(stats_score),
                    "final_score": float(final_score),
                    "is_anomaly": bool(is_anomaly),
                    "detection_method": detection_method,
                    "gt_label": type_anomaly != 'good'  # 正解ラベル（goodでなければ異常）
                }

                # Save the anomaly maps (raw as .npy or full resolution .tiff files)
                img_test_nr = img_test_nr.split(".")[0]
                if save_tiffs:
                    anomaly_map = dists2map(d_masked, image_test.shape)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.npy", d_masked)

                # Save some example plots (3 per anomaly type)
                if save_examples and idx < 3:

                    fig, (ax1, ax2, ax3, ax4,) = plt.subplots(1, 4, figsize=(18, 4.5))

                    # plot test image, PCA + mask
                    ax1.imshow(image_test)
                    ax2.imshow(vis_image_test_background)  

                    # plot patch distances 
                    d_masked[~mask2.reshape(grid_size2)] = 0.0
                    plt.colorbar(ax3.imshow(d_masked), ax=ax3, fraction=0.12, pad=0.05, orientation="horizontal")
                    
                    # compute image level anomaly score (mean(top 1%) of patches = empirical tail value at risk for quantile 0.99)
                    score_top1p = mean_top1p(distances)
                    ax4.axvline(score_top1p, color='r', linestyle='dashed', linewidth=1, label=round(score_top1p, 2))
                    ax4.legend()
                    ax4.hist(distances.flatten())

                    ax1.axis('off')
                    ax2.axis('off')
                    ax3.axis('off')

                    ax1.title.set_text("Test")
                    ax2.title.set_text("Test (PCA + Mask)")
                    ax3.title.set_text("Patch Distances (1NN)")
                    ax4.title.set_text("Hist of Distances")

                    plt.suptitle(f"Object: {object_name}, Type: {type_anomaly}, img = ...{image_test_path[-20:]}, object patches = {mask2.sum()}/{mask2.size}")

                    plt.tight_layout()
                    plt.savefig(f"{plots_dir}/{object_name}/examples/example_{type_anomaly}_{idx}.png")
                    plt.close()

    return anomaly_scores, time_memorybank, inference_times, detailed_scores