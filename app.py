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
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import torch.cuda.amp  # for mixed precision training

# Enable tensor cores for faster computation
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

# Initialize model with optimization flags
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda").eval()  # Ensure model is in eval mode
birefnet = torch.jit.script(birefnet)  # JIT compilation for faster inference

# Pre-compile transforms for better performance
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Increased batch size for better GPU utilization
BATCH_SIZE = 8  # Increased from 3
NUM_WORKERS = 4  # For parallel processing

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

def process_batch(batch_data):
    """Process a batch of frames in parallel"""
    images, backgrounds, image_sizes = zip(*batch_data)
    
    # Stack images for batch processing
    input_tensor = torch.stack(images).to("cuda")
    
    # Use automatic mixed precision for faster computation
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    
    processed_frames = []
    for pred, bg, size in zip(preds, backgrounds, image_sizes):
        mask = transforms.ToPILImage()(pred.squeeze()).resize(size)
        
        if isinstance(bg, str) and bg.startswith("#"):
            color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
            background = Image.new("RGBA", size, color_rgb + (255,))
        elif isinstance(bg, Image.Image):
            background = bg.convert("RGBA").resize(size)
        else:
            background = Image.open(bg).convert("RGBA").resize(size)
        
        # Use PIL's faster composite operation
        image = Image.composite(images[0].resize(size), background, mask)
        processed_frames.append(np.array(image))
    
    return processed_frames

@spaces.GPU
def fn(vid, bg_type="Color", bg_image=None, bg_video=None, color="#00FF00", fps=0, video_handling="slow_down"):
    try:
        # Load video more efficiently
        video = mp.VideoFileClip(vid, audio_buffersize=2000)
        if fps == 0:
            fps = video.fps
        audio = video.audio
        
        # Pre-calculate video parameters
        total_frames = int(video.fps * video.duration)
        frames = list(video.iter_frames(fps=fps))  # Load all frames at once
        
        # Pre-process background if using video
        if bg_type == "Video":
            bg_video_clip = mp.VideoFileClip(bg_video)
            if bg_video_clip.duration < video.duration:
                if video_handling == "slow_down":
                    bg_video_clip = bg_video_clip.fx(mp.vfx.speedx, 
                                                   factor=video.duration / bg_video_clip.duration)
                else:
                    multiplier = int(video.duration / bg_video_clip.duration + 1)
                    bg_video_clip = mp.concatenate_videoclips([bg_video_clip] * multiplier)
            background_frames = list(bg_video_clip.iter_frames(fps=fps))
        
        # Process frames in batches
        processed_frames = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i:i + BATCH_SIZE]
            batch_data = []
            
            for j, frame in enumerate(batch_frames):
                pil_image = Image.fromarray(frame)
                image_size = pil_image.size
                transformed_image = transform_image(pil_image)
                
                if bg_type == "Color":
                    bg = color
                elif bg_type == "Image":
                    bg = bg_image
                else:  # Video
                    frame_idx = (i + j) % len(background_frames)
                    bg = Image.fromarray(background_frames[frame_idx])
                
                batch_data.append((transformed_image, bg, image_size))
            
            # Process batch
            batch_results = process_batch(batch_data)
            processed_frames.extend(batch_results)
            
            # Yield progress updates
            if len(batch_results) > 0:
                yield batch_results[-1], None
        
        # Create output video
        processed_video = mp.ImageSequenceClip(processed_frames, fps=fps)
        if audio is not None:
            processed_video = processed_video.set_audio(audio)
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
            processed_video.write_videofile(output_path, codec="libx264", 
                                         preset='ultrafast', threads=NUM_WORKERS)
        
        yield gr.update(visible=False), gr.update(visible=True)
        yield processed_frames[-1], output_path
        
    except Exception as e:
        print(f"Error: {e}")
        yield gr.update(visible=False), gr.update(visible=True)
        yield None, f"Error processing video: {e}"


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    with gr.Row():
        in_video = gr.Video(label="Input Video", interactive=True)
        stream_image = gr.Image(label="Streaming Output", visible=False)
        out_video = gr.Video(label="Final Output Video")
    submit_button = gr.Button("Change Background", interactive=True)
    with gr.Row():
        fps_slider = gr.Slider(
            minimum=0,
            maximum=60,
            step=1,
            value=0,
            label="Output FPS (0 will inherit the original fps value)",
            interactive=True
        )
        bg_type = gr.Radio(["Color", "Image", "Video"], label="Background Type", value="Color", interactive=True)
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00", visible=True, interactive=True)
        bg_image = gr.Image(label="Background Image", type="filepath", visible=False, interactive=True)
        bg_video = gr.Video(label="Background Video", visible=False, interactive=True)
        with gr.Column(visible=False) as video_handling_options:
            video_handling_radio = gr.Radio(["slow_down", "loop"], label="Video Handling", value="slow_down", interactive=True)

    def update_visibility(bg_type):
        if bg_type == "Color":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif bg_type == "Image":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif bg_type == "Video":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


    bg_type.change(update_visibility, inputs=bg_type, outputs=[color_picker, bg_image, bg_video, video_handling_options])


    examples = gr.Examples(
        [
            ["rickroll-2sec.mp4", "Video", None, "background.mp4"],
            ["rickroll-2sec.mp4", "Image", "images.webp", None],
            ["rickroll-2sec.mp4", "Color", None, None],
        ],
        inputs=[in_video, bg_type, bg_image, bg_video],
        outputs=[stream_image, out_video],
        fn=fn,
        cache_examples=True,
        cache_mode="eager",
    )


    submit_button.click(
        fn,
        inputs=[in_video, bg_type, bg_image, bg_video, color_picker, fps_slider, video_handling_radio],
        outputs=[stream_image, out_video],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)