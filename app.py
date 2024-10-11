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

torch.set_float32_matmul_precision("highest")

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to("cuda")

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BATCH_SIZE = 3
executor = ThreadPoolExecutor(max_workers=4)

@spaces.GPU
def fn(vid, bg_type="Color", bg_image=None, bg_video=None, color="#00FF00", fps=0, video_handling="slow_down"):
    try:
        video = mp.VideoFileClip(vid)
        try:
            audio = video.audio
        except AttributeError:
            audio = None
        if fps == 0:
            fps = video.fps

        frames = video.iter_frames(fps=fps)
        processed_frames = []
        yield gr.update(visible=True), gr.update(visible=False)

        if bg_type == "Video":
            background_video = mp.VideoFileClip(bg_video)

            if background_video.duration < video.duration and video_handling == "slow_down":
                slow_down_factor = video.duration / background_video.duration
            else:
                slow_down_factor = 1
            background_frames = list(background_video.iter_frames(fps=fps))

        else:
            background_frames = None
            slow_down_factor = None


        bg_frame_index = 0
        frame_batch = []


        for i, frame in enumerate(frames):
            frame_batch.append(frame)

            if len(frame_batch) == BATCH_SIZE or i == int(video.fps * video.duration) - 1:

                pil_images = [Image.fromarray(f) for f in frame_batch]

                if bg_type == "Video":
                    processed_images = list(executor.map(process, pil_images, [get_background_image(bg_type, bg_image, background_frames, bg_frame_index + j, video_handling, slow_down_factor) for j in range(len(pil_images))]))
                    bg_frame_index += len(frame_batch)
                elif bg_type == "Color":
                    processed_images = list(executor.map(process, pil_images, [color] * len(pil_images)))
                elif bg_type == "Image":
                    processed_images = list(executor.map(process, pil_images, [bg_image] * len(pil_images)))
                else:
                    processed_images = pil_images

                for processed_image in processed_images:
                    processed_frames.append(np.array(processed_image))
                    yield processed_image, None
                frame_batch = []

        processed_video = mp.ImageSequenceClip(processed_frames, fps=fps)
        if audio:
            processed_video = processed_video.set_audio(audio)

        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = str(uuid.uuid4()) + ".mp4"
        temp_filepath = os.path.join(temp_dir, unique_filename)

        processed_video.write_videofile(temp_filepath, codec="libx264", logger=None)

        yield gr.update(visible=False), gr.update(visible=True)
        yield processed_image, temp_filepath

    except Exception as e:
        print(f"Error: {e}")
        yield gr.update(visible=False), gr.update(visible=True)
        yield None, f"Error processing video: {e}"


def process(image, bg):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    if isinstance(bg, str) and bg.startswith("#"):
        color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
        background = Image.new("RGBA", image_size, color_rgb + (255,))
    elif isinstance(bg, Image.Image):
        background = bg.convert("RGBA").resize(image_size)
    else:
        background = Image.open(bg).convert("RGBA").resize(image_size)

    # Composite the image onto the background using the mask
    image = Image.composite(image, background, mask)

    return image


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