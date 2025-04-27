# NightDayGAN: Image Domain Translation Using Pix2Pix

A TensorFlow 2 implementation of an advanced image-to-image translation model specifically designed for converting nighttime scenes to daytime imagery.

## Project Overview

NightDayGAN leverages the power of conditional Generative Adversarial Networks (cGANs) to perform realistic translations between night and day image domains. Based on the Pix2Pix architecture, this implementation is specifically optimized for the night2day dataset.

What makes our implementation unique:

- Efficient data preprocessing pipeline
- Enhanced U-Net generator architecture with skip connections
- PatchGAN discriminator for high-quality results
- Combined adversarial and L1 loss functions
- Comprehensive TensorBoard monitoring

## Quick Demo Examples

| Night Image   | Generated Day Image | Real Day Image |
| ------------- | ------------------- | -------------- |
| [Night Image] | [Generated Image]   | [Real Image]   |

## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- CUDA Toolkit (for GPU acceleration)
- Matplotlib
- NumPy
- FFmpeg (for video generation)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/nightdaygan.git
cd nightdaygan

# Install required packages
pip install -r requirements.txt
```

### Pre-trained Models

Download pre-trained models to use for inference or as a starting point for fine-tuning:

1. Create a directory where you can keep large files (ideally outside the project directory):

```bash
mkdir -p <pretrained_models>
```

2. Download pre-trained TF2 Saved Models from [Google Drive](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy)

3. Organize the downloaded files into the following structure:

```
<pretrained_models>/
├── film_net/
│   ├── L1/
│   ├── Style/
│   ├── VGG/
├── vgg/
│   ├── imagenet-vgg-verydeep-19.mat
```

### Dataset Preparation

1. Download the night2day dataset:

```bash
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz
tar xvf night2day.tar.gz
```

2. The dataset contains paired images of the same scene captured during night and day.

## Model Architecture

### Generator Design

Our generator uses a modified U-Net architecture:

- 8 downsampling layers (encoder)
- 7 upsampling layers with skip connections (decoder)
- Each layer uses 4×4 convolution kernels

### Discriminator Design

The discriminator employs a PatchGAN architecture:

- Classifies patches of the image as real or fake
- Takes concatenated input and target images
- Effective receptive field of 70×70 pixels

## Training

### Configuration

- Batch size: 1 (following best practices from the original Pix2Pix)
- Optimizer: Adam (learning rate: 2e-4, beta1: 0.5)
- L1 loss weight (lambda): 100
- Random augmentations: jitter, mirroring

### Running the Training

```python
# Start training with default parameters
python train.py

# With custom parameters
python train.py --batch_size 1 --epochs 200 --lambda 100
```

Training checkpoints are saved every 5,000 steps.

### Monitoring Progress

```python
# Launch TensorBoard
tensorboard --logdir logs/
```

## Inference

### Single Frame Generation

Generate a single middle frame between two input images:

```bash
# Generate a middle frame between two images
python -m eval.interpolator_test \
    --frame1 photos/one.png \
    --frame2 photos/two.png \
    --model_path pretrained_models/film_net/Style/saved_model \
    --output_frame photos/output_middle.png
```

### Multiple Frame Generation

Generate multiple interpolated frames and create a smooth video:

```bash
# Generate multiple frames and create a video
python -m eval.interpolator_cli \
    --pattern "photos" \
    --model_path pretrained_models/film_net/Style/saved_model \
    --times_to_interpolate 6 \
    --output_video
```

This will create interpolated frames in `photos/interpolated_frames/` and a video at `photos/interpolated.mp4`.

### Troubleshooting

If you encounter FFmpeg errors on Windows, please refer to this [FFmpeg Windows installation guide](https://phoenixnap.com/kb/ffmpeg-windows).

## Experimental Results

Our model achieves impressive results on the night2day test set:

- PSNR: _add your metrics here_
- SSIM: _add your metrics here_

Visual inspection shows realistic lighting conditions, natural colors, and preservation of important scene details.

## Comparison with Related Work

- Unlike traditional image enhancement methods, our approach learns the true mapping between domains
- Compared to CycleGAN, our paired training data allows for more precise translations
- Our implementation includes specific optimizations for the night-to-day domain conversion task

## Limitations

- Performance depends on the availability of aligned image pairs
- May struggle with extreme lighting conditions or unusual scenes not represented in the training data
- Some fine details may be lost in the translation process

## Acknowledgments

This project draws inspiration from the following papers:

- "Image-to-Image Translation with Conditional Adversarial Networks" by Isola et al.
- Various image translation literature in the computer vision community
