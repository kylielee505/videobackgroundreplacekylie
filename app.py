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
def fn(vid, fps=12, color="#00FF00"):
    # Load the video using moviepy
    video = mp.VideoFileClip(vid)

    # Extract audio from the video
    audio = video.audio

    # Extract frames at the specified FPS
    frames = video.iter_frames(fps=fps)

    # Process each frame for background removal
    processed_frames = []
    yield gr.update(visible=True), gr.update(visible=False)
    for frame in frames:
        pil_image = Image.fromarray(frame)
        processed_image = process(pil_image, color)
        processed_frames.append(np.array(processed_image))
        yield processed_image, None

    # Create a new video from the processed frames
    processed_video = mp.ImageSequenceClip(processed_frames, fps=fps)

    # Add the original audio back to the processed video
    processed_video = processed_video.set_audio(audio)

    # Save the processed video to a temporary file
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    unique_filename = str(uuid.uuid4()) + ".mp4"
    temp_filepath = os.path.join(temp_dir, unique_filename)
    processed_video.write_videofile(temp_filepath, codec="libx264")
    
    yield gr.update(visible=False), gr.update(visible=True)
    # Return the path to the temporary file
    yield None, temp_filepath


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
    color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))

    # Create a background image with the chosen color
    background = Image.new("RGBA", image_size, color_rgb + (255,))

    # Composite the image onto the background using the mask
    image = Image.composite(image, background, mask)

    return image


with gr.Blocks() as demo:
    with gr.Row():
        in_video = gr.Video(label="Input Video")
        stream_image = gr.Image(label="Streaming Output", visible=False)
        out_video = gr.Video(label="Final Output Video")  
    submit_button = gr.Button("Change Background")
    with gr.Row():
        fps_slider = gr.Slider(minimum=1, maximum=60, step=1, value=12, label="Output FPS")
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00")

    examples = gr.Examples(["rickroll-2sec.mp4"], inputs=in_video, outputs=[stream_image, out_video], fn=fn, cache_examples=True, cache_mode="eager")

    submit_button.click(
        fn, inputs=[in_video, fps_slider, color_picker], outputs=[stream_image, out_video]
    )
    
if __name__ == "__main__":
    demo.launch(show_error=True)