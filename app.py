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

    # Extract frames at the specified FPS
    frames = video.iter_frames(fps=fps)

    # Process each frame for background removal
    processed_frames = []
    yield gr.update(visible=True), gr.update(visible=False)
    for frame in frames:
        pil_image = Image.fromarray(frame)
        if bg_type == "Color":
            processed_image = process(pil_image, color)
        else:
            processed_image = process(pil_image, bg_image)
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
    yield processed_image, temp_filepath


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


with gr.Blocks() as demo:
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


    examples = gr.Examples(
        [("rickroll-2sec.mp4", "Image", "images.webp"), ("rickroll-2sec.mp4", "Color")],
        inputs=in_video,
        outputs=[stream_image, out_video],
        fn=fn,
        cache_examples=True,
        cache_mode="eager",
    )

    submit_button.click(
        fn,
        inputs=[in_video, bg_type, bg_image, color_picker, fps_slider],
        outputs=[stream_image, out_video],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)