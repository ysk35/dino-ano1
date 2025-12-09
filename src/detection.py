import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch
from sklearn.decomposition import PCA

from src.utils import augment_image, dists2map, plot_ref_images, apply_gamma_correction
from src.post_eval import mean_top1p


def apply_pca_whitening(features, n_components=256):
    """
    Apply PCA Whitening to features.
    This decorrelates features and normalizes variance, improving kNN distance.

    Args:
        features: numpy array of shape (n_samples, n_features)
        n_components: Number of PCA components to keep. Default 256.

    Returns:
        pca: Fitted PCA object (for transforming test features)
        transformed_features: Whitened features
    """
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    transformed = pca.fit_transform(features)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"    PCA Whitening: {features.shape[1]}D → {n_components}D (explained variance: {explained_var:.2%})")
    return pca, transformed.astype('float32')


def apply_coreset_subsampling(features, sampling_ratio=0.1, method='greedy'):
    """
    Apply coreset subsampling to reduce memory bank size while preserving coverage.

    Args:
        features: numpy array of shape (n_samples, n_features)
        sampling_ratio: Fraction of features to keep (0.0-1.0). Default 0.1 (10%)
        method: 'greedy' for greedy furthest point sampling, 'random' for random sampling

    Returns:
        selected_features: Subsampled features
        selected_indices: Indices of selected features
    """
    n_samples = features.shape[0]
    n_select = max(1, int(n_samples * sampling_ratio))

    if n_select >= n_samples:
        return features, np.arange(n_samples)

    if method == 'random':
        indices = np.random.choice(n_samples, n_select, replace=False)
        return features[indices], indices

    elif method == 'greedy':
        # Greedy furthest point sampling (minimax facility location)
        # This provides better coverage of feature space than random sampling
        selected_indices = []
        remaining = set(range(n_samples))

        # Start with random point
        first_idx = np.random.randint(n_samples)
        selected_indices.append(first_idx)
        remaining.remove(first_idx)

        # Compute distances from first point
        min_distances = np.linalg.norm(features - features[first_idx], axis=1)

        for _ in range(n_select - 1):
            if not remaining:
                break
            # Select point with maximum minimum distance to selected set
            remaining_list = list(remaining)
            furthest_idx = remaining_list[np.argmax(min_distances[remaining_list])]
            selected_indices.append(furthest_idx)
            remaining.remove(furthest_idx)

            # Update minimum distances
            new_distances = np.linalg.norm(features - features[furthest_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        selected_indices = np.array(selected_indices)
        print(f"    Coreset: {n_samples} → {len(selected_indices)} patches ({sampling_ratio:.0%})")
        return features[selected_indices], selected_indices

    else:
        raise ValueError(f"Unknown coreset method: {method}")


def apply_local_smoothing(distances_2d, kernel_size=3):
    """
    Apply local neighborhood smoothing to patch distances.
    This helps reduce noise and capture local context.

    Args:
        distances_2d: 2D array of patch distances (grid_size)
        kernel_size: Size of the smoothing kernel (default 3x3)

    Returns:
        Smoothed distances
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(distances_2d, size=kernel_size, mode='reflect')


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
        score_aggregation = 'mean_top1p',
        local_smoothing = False,
        smoothing_kernel = 3,
        use_pca_whitening = False,
        pca_components = 256,
        use_coreset = False,
        coreset_ratio = 0.1,
        coreset_method = 'greedy',
        gamma_value = 1.0,
        num_rotations = 8,
        use_multiscale = False,
        layers = None,
        layer_weights = None,
        normalize_distances = True):
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
    - score_aggregation: Method to aggregate patch scores. Options: 'mean_top1p', 'max', 'mean_top5p'. Default is 'mean_top1p'.
    - local_smoothing: Whether to apply local neighborhood smoothing. Default is False.
    - smoothing_kernel: Kernel size for local smoothing. Default is 3.
    - use_pca_whitening: Whether to apply PCA Whitening to features. Default is False.
    - pca_components: Number of PCA components. Default is 256.
    - use_coreset: Whether to apply coreset subsampling. Default is False.
    - coreset_ratio: Fraction of patches to keep (0.0-1.0). Default is 0.1.
    - coreset_method: 'greedy' or 'random'. Default is 'greedy'.
    - gamma_value: Gamma correction value. <1.0 brightens, >1.0 darkens. Default is 1.0 (no change).
    - num_rotations: Number of rotation augmentations. Default is 8.
    - use_multiscale: Whether to use multi-scale feature fusion. Default is False.
    - layers: List of layer indices for multiscale (e.g., [6, 12]). Required if use_multiscale is True.
    - layer_weights: Weights for each layer (must sum to 1). Required if use_multiscale is True.
    - normalize_distances: Whether to normalize distances from each layer before fusion. Default is True.
    """

    assert knn_metric in ["L2", "L2_normalized"]

    # Validate multiscale parameters
    if use_multiscale:
        assert layers is not None, "layers must be specified when use_multiscale is True"
        assert layer_weights is not None, "layer_weights must be specified when use_multiscale is True"
        assert len(layers) == len(layer_weights), "layers and layer_weights must have the same length"
        assert abs(sum(layer_weights) - 1.0) < 1e-6, "layer_weights must sum to 1"
        print(f"  Multiscale mode: layers={layers}, weights={layer_weights}, normalize={normalize_distances}")
    
    # add 'good' to the anomaly types
    type_anomalies = object_anomalies[object_name]
    type_anomalies.append('good')

    # ensure that each type is only evaluated once
    type_anomalies = list(set(type_anomalies))

    # Extract reference features
    features_ref = [] if not use_multiscale else {layer: [] for layer in layers}
    images_ref = []
    masks_ref = []
    vis_backgroud = []

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

            # Apply gamma correction if specified
            if gamma_value != 1.0:
                image_ref = apply_gamma_correction(image_ref, gamma_value)

            # augment reference image (if applicable)...
            if rotation:
                img_augmented = augment_image(image_ref, num_rotations=num_rotations)
            else:
                img_augmented = [image_ref]

            for i in range(len(img_augmented)):
                image_ref_aug = img_augmented[i]
                image_ref_tensor, grid_size1 = model.prepare_image(image_ref_aug)

                if use_multiscale:
                    # Extract features from multiple layers
                    features_dict = model.extract_features(image_ref_tensor, layers=layers)
                    # Use last layer for mask computation
                    features_for_mask = features_dict[layers[-1]]
                    mask_ref = model.compute_background_mask(features_for_mask, grid_size1, threshold=10, masking_type=(mask_ref_images and masking))
                    for layer in layers:
                        features_ref[layer].append(features_dict[layer][mask_ref])
                    if save_examples:
                        images_ref.append(image_ref_aug)
                        vis_image_background = model.get_embedding_visualization(features_for_mask, grid_size1, mask_ref)
                        masks_ref.append(mask_ref)
                        vis_backgroud.append(vis_image_background)
                else:
                    features_ref_i = model.extract_features(image_ref_tensor)
                    # compute background mask and discard background patches
                    mask_ref = model.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=(mask_ref_images and masking))
                    features_ref.append(features_ref_i[mask_ref])
                    if save_examples:
                        images_ref.append(image_ref_aug)
                        vis_image_background = model.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                        masks_ref.append(mask_ref)
                        vis_backgroud.append(vis_image_background)

        # Build kNN index (single or multi-scale)
        if use_multiscale:
            knn_indices = {}
            layer_norm_params = {}  # Store normalization parameters for each layer
            if not faiss_on_cpu:
                res = faiss.StandardGpuResources()
            for layer in layers:
                layer_features = np.concatenate(features_ref[layer], axis=0).astype('float32')
                if faiss_on_cpu:
                    knn_indices[layer] = faiss.IndexFlatL2(layer_features.shape[1])
                else:
                    knn_indices[layer] = faiss.GpuIndexFlatL2(res, layer_features.shape[1])
                if knn_metric == "L2_normalized":
                    faiss.normalize_L2(layer_features)
                knn_indices[layer].add(layer_features)
                print(f"    Layer {layer}: {layer_features.shape[0]} patches, dim={layer_features.shape[1]}")

            # Compute normalization parameters from memory bank (self-distance distribution)
            if normalize_distances:
                print("    Computing normalization parameters from memory bank...")
                for layer in layers:
                    # Sample features from memory bank to estimate distance distribution
                    n_samples = min(1000, knn_indices[layer].ntotal)
                    sample_indices = np.random.choice(knn_indices[layer].ntotal, n_samples, replace=False)

                    # Get sample features by reconstructing from index
                    layer_features_all = np.concatenate(features_ref[layer], axis=0).astype('float32')
                    if knn_metric == "L2_normalized":
                        faiss.normalize_L2(layer_features_all)
                    sample_features = layer_features_all[sample_indices]

                    # Search for 2-NN (exclude self which has distance 0)
                    distances, _ = knn_indices[layer].search(sample_features, k=2)
                    # Use second nearest neighbor (first is self with distance ~0)
                    nn_distances = distances[:, 1]
                    if knn_metric == "L2_normalized":
                        nn_distances = nn_distances / 2  # Convert to cosine distance
                    else:
                        nn_distances = np.sqrt(nn_distances)

                    # Compute mean and std for normalization
                    layer_norm_params[layer] = {
                        'mean': nn_distances.mean(),
                        'std': nn_distances.std()
                    }
                    print(f"    Layer {layer} norm params: mean={layer_norm_params[layer]['mean']:.4f}, std={layer_norm_params[layer]['std']:.4f}")
        else:
            features_ref = np.concatenate(features_ref, axis=0).astype('float32')
            print(f"  Memory bank: {features_ref.shape[0]} patches, {features_ref.shape[1]}D")

        # Apply PCA Whitening if enabled (single-scale only)
        pca_model = None
        if not use_multiscale:
            if use_pca_whitening:
                pca_model, features_ref = apply_pca_whitening(features_ref, n_components=pca_components)

            # Apply Coreset Subsampling if enabled
            if use_coreset:
                features_ref, _ = apply_coreset_subsampling(
                    features_ref, sampling_ratio=coreset_ratio, method=coreset_method
                )

            if faiss_on_cpu:
                # similarity search on CPU
                knn_index = faiss.IndexFlatL2(features_ref.shape[1])
            else:
                # similarity search on GPU
                res = faiss.StandardGpuResources()
                knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])

            if knn_metric == "L2_normalized":
                faiss.normalize_L2(features_ref)
            knn_index.add(features_ref)

        # end measuring time (for memory bank set up; in seconds, same for all test samples of this object)
        time_memorybank = time.time() - start_time

        # plot some reference samples for inspection
        if save_examples:
            plots_dir_ = f"{plots_dir}/{object_name}/"
            plot_ref_images(images_ref, masks_ref, vis_backgroud, grid_size1, plots_dir_, title = "Reference Images", img_names = img_ref_samples)   
        
        inference_times = {}
        anomaly_scores = {}

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

                # Apply gamma correction if specified
                if gamma_value != 1.0:
                    image_test = apply_gamma_correction(image_test, gamma_value)

                image_tensor2, grid_size2 = model.prepare_image(image_test)

                if use_multiscale:
                    # Extract features from multiple layers
                    features_dict2 = model.extract_features(image_tensor2, layers=layers)
                    features_for_mask2 = features_dict2[layers[-1]]

                    # Compute background mask using last layer
                    if masking:
                        mask2 = model.compute_background_mask(features_for_mask2, grid_size2, threshold=10, masking_type=masking)
                    else:
                        mask2 = np.ones(features_for_mask2.shape[0], dtype=bool)
                    if save_examples and idx < 3:
                        vis_image_test_background = model.get_embedding_visualization(features_for_mask2, grid_size2, mask2)

                    # Compute distances for each layer and fuse
                    fused_distances = np.zeros(mask2.sum(), dtype=float)
                    for layer_idx, layer in enumerate(layers):
                        layer_features = features_dict2[layer][mask2].astype('float32')

                        if knn_metric == "L2":
                            layer_distances, _ = knn_indices[layer].search(layer_features, k=knn_neighbors)
                            if knn_neighbors > 1:
                                layer_distances = layer_distances.mean(axis=1)
                            layer_distances = np.sqrt(layer_distances)
                        elif knn_metric == "L2_normalized":
                            faiss.normalize_L2(layer_features)
                            layer_distances, _ = knn_indices[layer].search(layer_features, k=knn_neighbors)
                            if knn_neighbors > 1:
                                layer_distances = layer_distances.mean(axis=1)
                            layer_distances = layer_distances / 2  # cosine distance

                        # Normalize distances if enabled
                        if normalize_distances and layer in layer_norm_params:
                            mean_d = layer_norm_params[layer]['mean']
                            std_d = layer_norm_params[layer]['std']
                            if std_d > 0:
                                layer_distances = (layer_distances - mean_d) / std_d

                        # Apply weight and accumulate
                        fused_distances += layer_weights[layer_idx] * layer_distances.squeeze()

                    output_distances = np.zeros_like(mask2, dtype=float)
                    output_distances[mask2] = fused_distances
                    distances = fused_distances  # For visualization
                else:
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

                    # Apply PCA transformation if enabled (using fitted model from training)
                    if use_pca_whitening and pca_model is not None:
                        features2 = pca_model.transform(features2).astype('float32')

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

                # Apply local smoothing if enabled
                if local_smoothing:
                    d_masked = apply_local_smoothing(d_masked, kernel_size=smoothing_kernel)
                    output_distances = d_masked.flatten()

                # Compute anomaly score with selected aggregation method
                if score_aggregation == 'mean_top1p':
                    final_score = mean_top1p(output_distances.flatten())
                elif score_aggregation == 'mean_top5p':
                    # Mean of top 5% - more robust to outliers
                    n_top = max(1, int(len(output_distances.flatten()) * 0.05))
                    final_score = np.mean(np.sort(output_distances.flatten())[-n_top:])
                elif score_aggregation == 'max':
                    final_score = np.max(output_distances.flatten())
                elif score_aggregation == 'median_top10':
                    # Median of top 10 distances - robust to outliers
                    final_score = np.median(np.sort(output_distances.flatten())[-10:])
                else:
                    final_score = mean_top1p(output_distances.flatten())

                # save inference time
                torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
                inf_time = time.time() - start_time
                inference_times[f"{type_anomaly}/{img_test_nr}"] = inf_time
                anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = final_score

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

    return anomaly_scores, time_memorybank, inference_times