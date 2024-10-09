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
def fn(vid, fps, color, progress=gr.Progress()):
    # Load the video using moviepy
    video = mp.VideoFileClip(vid)
    audio = video.audio
    frames = list(video.iter_frames(fps=fps))
    total_frames = len(frames)
    
    processed_frames_no_bg = []
    processed_frames_changed_bg = []
    
    # Create a live preview state
    preview_no_bg = None
    preview_with_bg = None
    
    for idx, frame in enumerate(progress.tqdm(frames)):
        pil_image = Image.fromarray(frame)
        processed_image, mask = process(pil_image, color)
        
        processed_frames_no_bg.append(np.array(processed_image))
        
        background = Image.new("RGBA", pil_image.size, color + (255,))
        composed_image = Image.composite(pil_image, background, mask)
        processed_frames_changed_bg.append(np.array(composed_image))
        
        # Update preview every 10 frames or on the last frame
        if idx % 10 == 0 or idx == total_frames - 1:
            preview_no_bg = np.array(processed_image)
            preview_with_bg = np.array(composed_image)
            yield preview_no_bg, preview_with_bg, None, None
    
    # Create videos from processed frames
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_video = mp.ImageSequenceClip(processed_frames_changed_bg, fps=fps)
    processed_video = processed_video.set_audio(audio)
    temp_filepath = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
    processed_video.write_videofile(temp_filepath, codec="libx264")
    
    processed_video_no_bg = mp.ImageSequenceClip(processed_frames_no_bg, fps=fps)
    processed_video_no_bg = processed_video_no_bg.set_audio(audio)
    temp_filepath_no_bg = os.path.join(temp_dir, f"{uuid.uuid4()}.webm")
    processed_video_no_bg.write_videofile(temp_filepath_no_bg, codec="libvpx")
    
    # Final yield with completed videos
    yield None, None, temp_filepath_no_bg, temp_filepath

def process(image, color_hex):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    background = Image.new("RGBA", image_size, color_rgb + (255,))
    image = Image.composite(image, background, mask)
    
    return image, mask

def change_color(in_video, fps_slider, color_picker):
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

with gr.Blocks() as demo:
    with gr.Row():
        in_video = gr.Video(label="Input Video")
        no_bg_video = gr.Video(label="No BG Video", visible=True)
        out_video = gr.Video(label="Output Video", visible=True)
    
    with gr.Row(visible=False) as preview_row:
        preview_no_bg = gr.Image(label="Live Preview (No Background)", visible=True)
        preview_with_bg = gr.Image(label="Live Preview (With Background)", visible=True)
    
    with gr.Row():
        fps_slider = gr.Slider(minimum=1, maximum=60, step=1, value=12, label="Output FPS")
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00")
    
    submit_button = gr.Button("Change Background")
    
    # Handle visibility changes and processing
    submit_button.click(
        fn=fn,
        inputs=[in_video, fps_slider, color_picker],
        outputs=[preview_no_bg, preview_with_bg, no_bg_video, out_video]
    )

if __name__ == "__main__":
    demo.launch(show_error=True)