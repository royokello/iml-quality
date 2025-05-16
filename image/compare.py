import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from typing import Union, Dict

def laplacian_filter_cuda(patch: torch.Tensor) -> torch.Tensor:
    # Move computation to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = torch.nn.functional.pad(patch.to(device), (1, 1, 1, 1), mode='reflect')
    return (4 * p[1:-1, 1:-1]
            - p[:-2, 1:-1] - p[2:, 1:-1]
            - p[1:-1, :-2] - p[1:-1, 2:])


def image_results(image_or_image_path: Union[str, os.PathLike, np.ndarray]) -> Dict[int, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(image_or_image_path, (str, os.PathLike)):
        # Load image and convert to tensor
        arr = torch.tensor(np.array(Image.open(image_or_image_path).convert("L")), dtype=torch.float32)
    elif isinstance(image_or_image_path, np.ndarray):
        arr = torch.tensor(image_or_image_path, dtype=torch.float32)
        if arr.ndim == 3:
            # Convert RGB to grayscale using GPU if available
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
            arr = torch.tensordot(arr[..., :3], weights, dims=([2], [0]))
    else:
        raise TypeError("Expected file path or ndarray")

    h, w = arr.shape
    results: Dict[int, float] = {}
    
    # Process patches in parallel using GPU
    patches = arr.unfold(0, 8, 8).unfold(1, 8, 8)  # Shape: (H//8, W//8, 8, 8)
    patches = patches.reshape(-1, 8, 8)  # Reshape to (N, 8, 8) where N = (H//8 * W//8)
    
    # Move to GPU and compute Laplacian for all patches at once
    lap_results = laplacian_filter_cuda(patches)
    variances = torch.var(lap_results, dim=(1,2))
    
    # Move results back to CPU and convert to dictionary
    variances_cpu = variances.cpu().numpy()
    results = {i: float(var) for i, var in enumerate(variances_cpu)}
    
    return results


def get_size(image_or_path: Union[str, os.PathLike, np.ndarray]) -> int:
    if isinstance(image_or_path, (str, os.PathLike)):
        return os.path.getsize(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        return image_or_path.nbytes
    else:
        raise TypeError("Expected file path or ndarray")


def compare_images(
    image1: Union[str, os.PathLike, np.ndarray],
    image2: Union[str, os.PathLike, np.ndarray],
) -> Dict[str, float]:
    scores1 = list(image_results(image1).values())
    scores2 = list(image_results(image2).values())
    avg1 = float(np.mean(scores1)) if scores1 else 0.0
    avg2 = float(np.mean(scores2)) if scores2 else 0.0
    diff_banding = abs(avg2 - avg1)

    size1 = get_size(image1)
    size2 = get_size(image2)
    size_diff = float(abs(size2 - size1))

    return {
        "image_1_avg_banding": avg1,
        "image_2_avg_banding": avg2,
        "avg_banding_diff": diff_banding,
        "storage_size_diff": size_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two images by average banding and file size."
    )
    parser.add_argument("image1", help="Path to first image file")
    parser.add_argument("image2", help="Path to second image file")
    args = parser.parse_args()

    results = compare_images(args.image1, args.image2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
