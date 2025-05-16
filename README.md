# iml-quality
A Python tool to quantify and analyze image quality using advanced metrics.

## Features
- **Image Quality Assessment**: Evaluates images based on multiple quality metrics including:
  - Resolution analysis
  - Aspect ratio optimization
  - Banding detection
  - Edge detail analysis

- **Image Comparison**: Compare two images based on:
  - Average banding scores
  - Storage size differences
  - Edge detail preservation

- **Batch Processing**: Extract top-quality images from a directory based on:
  - Configurable selection modes (percentile or fixed count)
  - Customizable quality thresholds
  - Automatic duplicate handling

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/iml-quality.git
cd iml-quality

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Image Comparison
Compare two images to analyze their quality differences:
```bash
python -m image.compare image1.jpg image2.jpg
```

### Extract Top Quality Images
Extract the highest quality images from a directory:
```bash
python -m image.extract --input /path/to/images --output /path/to/output --mode percentile --amount 0.15
```

Parameters:
- `--input`: Source directory containing images
- `--output`: Directory to save selected images
- `--mode`: Selection mode ('percentile' or 'fixed')
- `--amount`: Amount to extract (0.15 for top 15% in percentile mode, or specific number in fixed mode)

## Technical Details

### Quality Metrics
The tool uses several metrics to assess image quality:
1. Resolution score: Based on image dimensions
2. Aspect ratio analysis: Penalizes extreme aspect ratios
3. Banding detection: Uses Laplacian filtering to detect and quantify banding artifacts
4. Edge detail preservation: Analyzes image sharpness and detail retention

### GPU Acceleration
The tool utilizes GPU acceleration (CUDA) when available for faster processing of large image sets.

## Requirements
- Python 3.6+
- PyTorch
- Pillow (PIL)
- NumPy
- CUDA-capable GPU (optional, for acceleration)

## License
[Add your license information here]
