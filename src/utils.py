import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def preprocess_image(img, method="none", **kwargs):
    """
    Apply image preprocessing techniques for anomaly detection.

    Parameters:
    - img: Input image (RGB format, numpy array)
    - method: Preprocessing method to apply (string or list of strings)
        - "none": No preprocessing (default)
        - "clahe": Contrast Limited Adaptive Histogram Equalization
        - "gamma": Gamma correction
        - "sharpening": Image sharpening
        - "clamp": Clamp pixel values to specified range
        - Can also be a list like ["gamma", "clahe"] to apply multiple in order
        - Or a string like "gamma+clahe" which will be split and applied in order
    - kwargs: Additional parameters for specific methods
        - clahe: clip_limit (default=2.0), tile_grid_size (default=8)
        - gamma: gamma_value (default=1.0)
        - clamp: min_value (default=0), max_value (default=255)

    Returns:
    - Preprocessed image (same shape as input)
    """
    # Handle combination of methods
    if isinstance(method, str) and "+" in method:
        methods = method.split("+")
    elif isinstance(method, list):
        methods = method
    else:
        methods = [method]

    # Apply each method in sequence
    result = img
    for m in methods:
        result = _apply_single_preprocess(result, m, **kwargs)

    return result


def _apply_single_preprocess(img, method, **kwargs):
    """Apply a single preprocessing method."""
    if method == "none":
        return img

    elif method == "clahe":
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_grid_size = kwargs.get("tile_grid_size", 8)

        # Convert RGB to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return img_clahe

    elif method == "gamma":
        gamma_value = kwargs.get("gamma_value", 1.0)

        # Normalize to [0, 1], apply gamma correction, then scale back
        img_gamma = np.power(img / 255.0, gamma_value) * 255.0
        return img_gamma.astype(np.uint8)

    elif method == "sharpening":
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        img_sharp = cv2.filter2D(img, -1, kernel)
        return img_sharp

    elif method == "clamp":
        min_value = kwargs.get("min_value", 0)
        max_value = kwargs.get("max_value", 255)

        # Clamp pixel values to [min_value, max_value]
        img_clamped = np.clip(img, min_value, max_value)
        return img_clamped.astype(np.uint8)

    elif method == "histeq":
        # Histogram equalization on L channel (LAB color space)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif method == "bilateral":
        # Bilateral filter: noise reduction while preserving edges
        d = kwargs.get("bilateral_d", 9)  # Diameter of pixel neighborhood
        sigma_color = kwargs.get("bilateral_sigma_color", 75)
        sigma_space = kwargs.get("bilateral_sigma_space", 75)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    elif method == "gaussian":
        # Gaussian blur for noise reduction
        kernel_size = kwargs.get("gaussian_kernel", 5)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    elif method == "unsharp":
        # Unsharp masking: sharpen by subtracting blurred version
        amount = kwargs.get("unsharp_amount", 1.5)
        kernel_size = kwargs.get("unsharp_kernel", 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    elif method == "denoise":
        # Non-local means denoising
        h = kwargs.get("denoise_h", 10)  # Filter strength
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    elif method == "normalize":
        # Per-channel normalization to [0, 255]
        result = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            channel = img[:, :, i].astype(np.float32)
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                result[:, :, i] = (channel - min_val) / (max_val - min_val) * 255
            else:
                result[:, :, i] = channel
        return result.astype(np.uint8)

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def augment_image(img_ref, augmentation = "rotate", angles = None, num_rotations = 8):
    """
    Data augmentation for images, supporting flexible rotation.

    Parameters:
    - img_ref: Reference image (numpy array)
    - augmentation: Augmentation method (default: "rotate")
    - angles: List of rotation angles. If None, generates evenly spaced angles
    - num_rotations: Number of rotations to generate (default: 8)
                     Common values: 8, 16, 32 for increasing memory bank density

    Returns:
    - List of augmented images
    """
    imgs = []
    if augmentation == "rotate":
        # If angles not provided, generate evenly spaced angles
        if angles is None:
            angles = [i * (360 / num_rotations) for i in range(num_rotations)]

        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    return result


def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    mask = np.repeat(mask, imgd2, axis=1)
    return mask


def plot_ref_images(img_list, mask_list, vis_background_list, grid_size, save_path, title = "Reference Images", img_names = None):
    k = min(len(img_list), 32)  # reduce max number of ref samples to plot to 32

    n_aug = len(img_list)//len(img_names)

    fig, axs = plt.subplots(k, 3, figsize=(10, 3.5*k))
    if k == 1:
        axs = axs.reshape(1, -1)
    for i in range(k):
        axs[i, 0].imshow(img_list[i])
        axs[i, 1].imshow(vis_background_list[i])
        axs[i, 2].imshow(img_list[i])
        axs[i, 2].imshow(resize_mask_img(mask_list[i], img_list[i].shape, grid_size), alpha=0.5)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        if i % n_aug == 0:
            axs[i, 0].title.set_text(f"Image: {img_names[i // n_aug]}")
        else:
            axs[i, 0].title.set_text(f"Augmentation of Image {img_names[i // n_aug]}")
        axs[i, 1].title.set_text("PCA + Mask")
        axs[i, 2].title.set_text("Mask")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "reference_samples.png")
    plt.close()


def get_dataset_info(dataset, preprocess):

    if preprocess not in ["informed", "agnostic", "masking_only", "informed_no_mask", "agnostic_no_mask", "force_no_mask_no_rotation", "force_mask_no_rotation", "force_no_mask_rotation", "force_mask_rotation"]:
        # masking only: deactivate rotation, apply masking like in informed/agnostic
        raise ValueError(f"Preprocessing '{preprocess}' not yet covered!")
    
    if dataset == "MVTec":
        objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        object_anomalies = {"bottle": ["broken_large", "broken_small", "contamination"],
                            "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_wire", "missing_cable", "poke_insulation"],
                            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
                            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
                            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
                            "hazelnut": ["crack", "cut", "hole", "print"],
                            "leather": ["color", "cut", "fold", "glue", "poke"],
                            "metal_nut": ["bent", "color", "flip", "scratch"],
                            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
                            "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
                            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
                            "toothbrush": ["defective"], 
                            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
                            "wood": ["color", "combined", "hole", "liquid", "scratch"],
                            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]
                            }

        if preprocess in ["agnostic", "informed", "masking_only"]:
            # Define Masking for the different objects -> determine with Masking Test (see Fig. 2 and discussion in the paper)
            # True: default masking (threshold the first PCA component > 10)
            # False: No masking will be applied
            masking_default = {"bottle": False,      
                                "cable": False,         # no masking
                                "capsule": True,        # default masking
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,
                                "leather": False,
                                "metal_nut": False,
                                "pill": True,
                                "screw": True,
                                "tile": False,
                                "toothbrush": True,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }
            
        if preprocess in ["informed", "informed_no_mask"]:
            rotation_default = {"bottle": False,
                                "cable": False, 
                                "capsule": False,
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,       # informed: hazelnut is rotated
                                "leather": False,
                                "metal_nut": False,
                                "pill": False,          # informed: all pills in train are oriented just the same
                                "screw": True,          # informed: screws in train are oriented differently
                                "tile": False,
                                "toothbrush": False,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }

        elif preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess == "masking_only":
            rotation_default = {o: False for o in objects}

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}

    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        object_anomalies = {"candle": ["bad"],
                            "capsules": ["bad"],
                            "cashew": ["bad"],
                            "chewinggum": ["bad"],
                            "fryum": ["bad"],
                            "macaroni1": ["bad"],
                            "macaroni2": ["bad"],
                            "pcb1": ["bad"],
                            "pcb2": ["bad"],
                            "pcb3": ["bad"],
                            "pcb4": ["bad"],
                            "pipe_fryum": ["bad"],
                            }

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}

        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}
    else:
        raise ValueError(f"Dataset '{dataset}' not yet covered!")

    if preprocess == "force_no_mask_no_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_mask_no_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_no_mask_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: True for o in objects}
    elif preprocess == "force_mask_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: True for o in objects}

    return objects, object_anomalies, masking_default, rotation_default