FILM: Frame Interpolation for Large Motion
Official TensorFlow 2 implementation of high-quality frame interpolation without pre-trained optical flow or depth models.
Presented at ECCV 2022.

Install dependencies:

pip install -r requirements.txt
sudo apt-get install -y ffmpeg
⚡ Requirements: Python 3.9, TensorFlow 2.x, CUDA 11.2.1, cuDNN 8.1.0 if using GPU.

Download Pre-trained Models
mkdir -p <pretrained_models>
Download pre-trained models from https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy

Place folders as:

<pretrained_models>/
├── film_net/
│   ├── L1/
│   ├── Style/
│   ├── VGG/
├── vgg/
│   └── imagenet-vgg-verydeep-19.mat



Quick Start
Generate an intermediate frame between two images:

python3 -m eval.interpolator_test \
    --frame1 photos/one.png \
    --frame2 photos/two.png \
    --model_path <pretrained_models>/film_net/Style/saved_model \
    --output_frame photos/output_middle.png

    
Generate many intermediate frames and create a video:

python3 -m eval.interpolator_cli \
    --pattern "photos" \
    --model_path <pretrained_models>/film_net/Style/saved_model \
    --times_to_interpolate 6 \
    --output_video
Output: Interpolated frames at photos/interpolated_frames/ and a video at photos/interpolated.mp4.


Training (Optional)
Train from scratch (TFRecord input required):

python3 -m training.train \
    --gin_config training/config/film_net-Style.gin \
    --base_folder <path_to_save_training> \
    --label <run_name>
Evaluate (Optional)
Evaluate on a benchmark dataset:

python3 -m eval.eval_cli \
    --gin_config eval/config/middlebury.gin \
    --model_path <pretrained_models>/film_net/L1/saved_model
