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
def fn(vid, bg_type="Color", bg_image=None, color="#00FF00", fps=0):
    # Load the video using moviepy
    video = mp.VideoFileClip(vid)

    # Load original fps if fps value is equal to 0
    if fps == 0:
        fps = video.fps

    # Extract audio from the video
    audio = video.audio

    # Process video in chunks of 1 second
    chunk_duration = 1  # seconds
    total_duration = video.duration
    start_time = 0
        
    progress = f'<div class="progress-container"><div class="progress-bar" style="--current: {start_time}; --total: {total_duration};"></div></div>'

    processed_frames = []
    yield gr.update(visible=True), gr.update(visible=False), progress

    while start_time < total_duration:
        end_time = min(start_time + chunk_duration, total_duration)
        chunk = video.subclip(start_time, end_time)
        chunk_frames = chunk.iter_frames(fps=fps)

        for frame in chunk_frames:
            pil_image = Image.fromarray(frame)
            if bg_type == "Color":
                processed_image = process(pil_image, color)
            else:
                processed_image = process(pil_image, bg_image)
            processed_frames.append(np.array(processed_image))
            yield processed_image, None, progress

        # Save processed frames for the current chunk
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        for i, frame in enumerate(processed_frames):
            Image.fromarray(frame).save(os.path.join(temp_dir, f"frame_{start_time}_{i}.png"))

        # Clear processed frames for the current chunk
        processed_frames = []
        progress = f'<div class="progress-container"><div class="progress-bar" style="--current: {start_time}; --total: {total_duration};"></div></div>'

        yield None, None, progress

        start_time += chunk_duration

    # Load all saved frames
    all_frames = []
    for filename in sorted(os.listdir(temp_dir)):
        if filename.startswith("frame_") and filename.endswith(".png"):
            frame = np.array(Image.open(os.path.join(temp_dir, filename)))
            all_frames.append(frame)

    # Create a new video from the processed frames
    processed_video = mp.ImageSequenceClip(all_frames, fps=fps)

    # Add the original audio back to the processed video
    processed_video = processed_video.set_audio(audio)

    # Save the processed video to a temporary file
    temp_filepath = os.path.join(temp_dir, "processed_video.mp4")
    processed_video.write_videofile(temp_filepath, codec="libx264")

    # Clean up temporary files
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))

    yield gr.update(visible=False), gr.update(visible=True), progress
    # Return the path to the temporary file
    yield processed_image, temp_filepath, progress


def process(image, bg):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    if bg.startswith("#"):
        color_rgb = tuple(int(bg[i : i + 2], 16) for i in (1, 3, 5))
        background = Image.new("RGBA", image_size, color_rgb + (255,))
    else:
        background = Image.open(bg).convert("RGBA").resize(image_size)

    # Composite the image onto the background using the mask
    image = Image.composite(image, background, mask)

    return image


css="""
.progress-container {width: 100%;height: 30px;background-color: #f0f0f0;border-radius: 15px;overflow: hidden;margin-bottom: 20px}
.progress-bar {height: 100%;background-color: #4f46e5;width: calc(var(--current) / var(--total) * 100%);transition: width 0.5s ease-in-out}
"""

with gr.Blocks(css=css, theme="ocean") as demo:
    with gr.Row():
        in_video = gr.Video(label="Input Video")
        stream_image = gr.Image(label="Streaming Output", visible=False)
        out_video = gr.Video(label="Final Output Video")
    submit_button = gr.Button("Change Background")
    with gr.Row():
        fps_slider = gr.Slider(
            minimum=0,
            maximum=60,
            step=1,
            value=0,
            label="Output FPS (0 will inherit the original fps value)",
        )
        bg_type = gr.Radio(["Color", "Image"], label="Background Type", value="Color")
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00", visible=True)
        bg_image = gr.Image(label="Background Image", type="filepath", visible=False)

    def update_visibility(bg_type):
        if bg_type == "Color":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    bg_type.change(update_visibility, inputs=bg_type, outputs=[color_picker, bg_image])

    progress_bar = gr.Markdown(elem_id="progress")

    examples = gr.Examples(
        [["rickroll-2sec.mp4", "Image", "images.webp"], ["rickroll-2sec.mp4", "Color", None]],
        inputs=[in_video, bg_type, bg_image],
        outputs=[stream_image, out_video, progress_bar],
        fn=fn,
        cache_examples=True,
        cache_mode="eager",
    )


    submit_button.click(
        fn,
        inputs=[in_video, bg_type, bg_image, color_picker, fps_slider],
        outputs=[stream_image, out_video, progress_bar],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)