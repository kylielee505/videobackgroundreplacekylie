import gradio as gr
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import moviepy.editor as mp
from pydub import AudioSegment
from PIL import Image
import numpy as np
import os
import tempfile
import uuid

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@spaces.GPU
def fn(vid, fps, color):
    # Load the video using moviepy
    video = mp.VideoFileClip(vid)

    # Extract audio from the video
    audio = video.audio

    # Extract frames at the specified FPS
    frames = video.iter_frames(fps=fps)

    # Process each frame for background removal
    processed_frames_no_bg = []
    processed_frames_changed_bg = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        processed_image, mask = process(pil_image, color)  # Get both processed image and mask
        processed_frames_no_bg.append(np.array(processed_image))  # Save no-background frame
        
        # Compose with background for changed background video
        background = Image.new("RGBA", pil_image.size, color + str((255,)))
        composed_image = Image.composite(pil_image, background, mask)
        processed_frames_changed_bg.append(np.array(composed_image))

    # Create a new video from the processed frames
    processed_video = mp.ImageSequenceClip(processed_frames_changed_bg, fps=fps)

    # Add the original audio back to the processed video
    processed_video = processed_video.set_audio(audio)

    # Save the processed video to a temporary file
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    unique_filename = str(uuid.uuid4()) + ".mp4"
    temp_filepath = os.path.join(temp_dir, unique_filename)
    processed_video.write_videofile(temp_filepath, codec="libx264")

    # Create and save no-background video
    processed_video_no_bg = mp.ImageSequenceClip(processed_frames_no_bg, fps=fps)
    processed_video_no_bg = processed_video_no_bg.set_audio(audio)
    temp_filepath_no_bg = os.path.join(temp_dir, str(uuid.uuid4()) + ".webm")
    processed_video_no_bg.write_videofile(temp_filepath_no_bg, codec="libvpx")

    return temp_filepath_no_bg, temp_filepath


def process(image, color_hex):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    # Convert hex color to RGB tuple
    color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))

    # Create a background image with the chosen color
    background = Image.new("RGBA", image_size, color_rgb + (255,))

    # Composite the image onto the background using the mask
    image = Image.composite(image, background, mask)

    return image, mask  # Return both the processed image and the mask


with gr.Blocks() as demo:
    with gr.Row():
        in_video = gr.Video(label="Input Video")
        no_bg_video = gr.Video(label="No BG Video")  # Added for no-background video
        out_video = gr.Video(label="Output Video")  # This will be the changed-background video
    submit_button = gr.Button("Change Background")
    with gr.Row():
        fps_slider = gr.Slider(minimum=1, maximum=60, step=1, value=12, label="Output FPS")
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00")


    submit_button.click(
        fn, inputs=[in_video, fps_slider, color_picker], outputs=[no_bg_video, out_video] 
    )

if __name__ == "__main__":
    demo.launch(show_error=True)