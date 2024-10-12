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
import schedule
import time
import threading
import shutil

torch.set_float32_matmul_precision("medium")

device = "cuda" if torch.cuda.is_available() else "cpu"

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(device)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_background_image(bg, image_size):
    """Loads and resizes the background image based on the provided input."""
    try:
        if isinstance(bg, str) and bg.startswith("#"):
            color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
            return Image.new("RGBA", image_size, color_rgb + (255,))
        elif isinstance(bg, Image.Image):
            return bg.convert("RGBA").resize(image_size)
        else:
            return Image.open(bg).convert("RGBA").resize(image_size)
    except Exception as e:
        print(f"Error opening background image: {e}")
        return None

def clear_temp_directory():
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def run_scheduler():
    schedule.every(10).minutes.do(clear_temp_directory)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True  # Allow the main thread to exit even if the scheduler is running
scheduler_thread.start()

@spaces.GPU
def process_video(input_video_path, bg_type="Color", bg_image_path=None, bg_video_path=None, 
                  bg_color="#00FF00", output_fps=0, video_handling_mode="slow_down"):
    """Processes the input video and replaces the background."""
    try:
        # Load the video using moviepy
        video = mp.VideoFileClip(input_video_path)

        # Load original fps if fps value is equal to 0
        if output_fps == 0:
            output_fps = video.fps

        # Extract audio from the video
        audio = video.audio

        # Extract frames at the specified FPS
        frames = video.iter_frames(fps=output_fps)

        # Process each frame for background removal
        processed_frames = []
        yield gr.update(visible=True), gr.update(visible=False), gr.update(value=0)

        if bg_type == "Video":
            background_video = mp.VideoFileClip(bg_video_path)
            if background_video.duration < video.duration:
                if video_handling_mode == "slow_down":
                    background_video = background_video.fx(mp.vfx.speedx, factor=video.duration / background_video.duration)
                else:  # video_handling_mode == "loop"
                    background_video = mp.concatenate_videoclips([background_video] * int(video.duration / background_video.duration + 1))
            background_frames = list(background_video.iter_frames(fps=output_fps))  # Convert to list
        else:
            background_frames = None

        bg_frame_index = 0  # Initialize background frame index
        total_frames = len(list(frames))
        frames = video.iter_frames(fps=output_fps)
        batch_size = 4 

        for i in range(0, total_frames, batch_size):
            batch_frames = list(frames)[i:i+batch_size]
            processed_batch = []
            for frame in batch_frames:
                pil_image = Image.fromarray(frame)
                if bg_type == "Color":
                    background_image = load_background_image(bg_color, pil_image.size)
                elif bg_type == "Image":
                    background_image = load_background_image(bg_image_path, pil_image.size)
                elif bg_type == "Video":
                    if video_handling_mode == "slow_down":
                        background_frame = background_frames[bg_frame_index % len(background_frames)]
                        bg_frame_index += 1
                        background_image = Image.fromarray(background_frame)
                    else:  # video_handling_mode == "loop"
                        background_frame = background_frames[bg_frame_index % len(background_frames)]
                        bg_frame_index += 1
                        background_image = Image.fromarray(background_frame)
                else:
                    background_image = None  # Default to original image if no background is selected

                if background_image is not None:
                    processed_image = process(pil_image, background_image)
                else:
                    processed_image = pil_image

                processed_batch.append(np.array(processed_image))

            processed_frames.extend(processed_batch)
            progress = (i + len(batch_frames)) / total_frames
            yield processed_batch[-1], None, gr.update(value=progress)  # Update progress bar

        # Create a new video from the processed frames
        processed_video = mp.ImageSequenceClip(processed_frames, fps=output_fps)

        # Add the original audio back to the processed video
        processed_video = processed_video.set_audio(audio)

        # Save the processed video to a temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = str(uuid.uuid4()) + ".mp4"
        temp_filepath = os.path.join(temp_dir, unique_filename)
        processed_video.write_videofile(temp_filepath, codec="libx264")

        yield gr.update(visible=False), gr.update(visible=True), gr.update(value=1)
        # Return the path to the temporary file
        yield processed_batch[-1], temp_filepath

    except Exception as e:
        error_message = f"Error processing video: {e}"
        print(error_message)
        yield gr.update(visible=False), gr.update(visible=True), gr.update(value=0)
        yield None, error_message


def process(image, bg):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    # Composite the image onto the background using the mask
    image = Image.composite(image, bg, mask)

    return image


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Video Background Remover & Changer\n### You can replace image background with any color, image or video.\nNOTE: As this Space is running on ZERO GPU it has limit. It can handle approx 200frmaes at once. So, if you have big video than use small chunks or Duplicate this space.")
    with gr.Row():
        in_video = gr.Video(label="Input Video", interactive=True)
        stream_image = gr.Image(label="Streaming Output", visible=False)
        out_video = gr.Video(label="Final Output Video")
    submit_button = gr.Button("Change Background", interactive=True)
    progress_bar = gr.ProgressBar(label="Processing Progress")
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
        outputs=[stream_image, out_video, progress_bar],
        fn=process_video,
        cache_examples=True,
        cache_mode="eager",
    )


    submit_button.click(
        process_video,
        inputs=[in_video, bg_type, bg_image, bg_video, color_picker, fps_slider, video_handling_radio],
        outputs=[stream_image, out_video, progress_bar],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)