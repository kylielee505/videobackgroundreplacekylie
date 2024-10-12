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
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

torch.set_float32_matmul_precision("medium")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both BiRefNet models
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(device)

birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet_lite", trust_remote_code=True
)
birefnet_lite.to(device)

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Function to delete files older than 10 minutes in the temp directory
def cleanup_temp_files():
    while True:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath):
                    file_age = time.time() - os.path.getmtime(filepath)
                    if file_age > 600:  # 10 minutes in seconds
                        try:
                            os.remove(filepath)
                            print(f"Deleted temporary file: {filepath}")
                        except Exception as e:
                            print(f"Error deleting file {filepath}: {e}")
        time.sleep(60)  # Check every minute


# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_temp_files, daemon=True)
cleanup_thread.start()

def process(image, bg, fast_mode=False):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Select the model based on fast_mode
    model = birefnet_lite if fast_mode else birefnet

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
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


@spaces.GPU
def fn(vid, bg_type="Color", bg_image=None, bg_video=None, color="#00FF00", fps=0, video_handling="slow_down", fast_mode=True):
    try:
        start_time = time.time()  # Start the timer

        # Load the video using moviepy
        video = mp.VideoFileClip(vid)

        # Load original fps if fps value is equal to 0
        if fps == 0:
            fps = video.fps

        # Extract audio from the video
        audio = video.audio

        # Extract frames at the specified FPS
        frames = list(video.iter_frames(fps=fps))

        # Process frames in parallel
        processed_frames = []
        yield gr.update(visible=True), gr.update(visible=False), "Processing started... Elapsed time: 0 seconds"

        if bg_type == "Video":
            background_video = mp.VideoFileClip(bg_video)
            if background_video.duration < video.duration:
                if video_handling == "slow_down":
                    background_video = background_video.fx(mp.vfx.speedx, factor=video.duration / background_video.duration)
                else:  # video_handling == "loop"
                    background_video = mp.concatenate_videoclips([background_video] * int(video.duration / background_video.duration + 1))
            background_frames = list(background_video.iter_frames(fps=fps))  # Convert to list
        else:
            background_frames = None

        bg_frame_index = 0  # Initialize background frame index

        # Define a helper function for processing a single frame
        def process_single_frame(i, frame):
            pil_image = Image.fromarray(frame)
            if bg_type == "Color":
                processed_image = process(pil_image, color, fast_mode)
            elif bg_type == "Image":
                processed_image = process(pil_image, bg_image, fast_mode)
            elif bg_type == "Video":
                if video_handling == "slow_down":
                    background_frame = background_frames[bg_frame_index % len(background_frames)]
                else:  # video_handling == "loop"
                    background_frame = background_frames[bg_frame_index % len(background_frames)]
                nonlocal bg_frame_index
                bg_frame_index += 1
                background_image = Image.fromarray(background_frame)
                processed_image = process(pil_image, background_image, fast_mode)
            else:
                processed_image = pil_image  # Default to original image if no background is selected
            return i, np.array(processed_image)

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all frame processing tasks
            future_to_index = {executor.submit(process_single_frame, i, frame): i for i, frame in enumerate(frames)}
            
            # As each future completes, process the result
            for future in as_completed(future_to_index):
                i, processed_image = future.result()
                processed_frames.append((i, processed_image))
                
                # Update elapsed time
                elapsed_time = time.time() - start_time
                # Sort the processed_frames based on index to maintain order
                processed_frames_sorted = sorted(processed_frames, key=lambda x: x[0])
                
                # Yield the first processed image if it's available
                if len(processed_frames_sorted) == 1:
                    first_image = Image.fromarray(processed_frames_sorted[0][1])
                    yield first_image, None, f"Processing frame {processed_frames_sorted[0][0]+1}... Elapsed time: {elapsed_time:.2f} seconds"
        
        # Sort all processed frames
        processed_frames_sorted = sorted(processed_frames, key=lambda x: x[0])
        final_frames = [frame for i, frame in processed_frames_sorted]

        # Create a new video from the processed frames
        processed_video = mp.ImageSequenceClip(final_frames, fps=fps)

        # Add the original audio back to the processed video
        processed_video = processed_video.set_audio(audio)

        # Save the processed video to a temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = str(uuid.uuid4()) + ".mp4"
        temp_filepath = os.path.join(temp_dir, unique_filename)
        processed_video.write_videofile(temp_filepath, codec="libx264")

        elapsed_time = time.time() - start_time
        yield gr.update(visible=False), gr.update(visible=True), f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds"
        # Return the path to the temporary file
        yield None, temp_filepath, f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds"

    except Exception as e:
        print(f"Error: {e}")
        elapsed_time = time.time() - start_time
        yield gr.update(visible=False), gr.update(visible=True), f"Error processing video: {e}. Elapsed time: {elapsed_time:.2f} seconds"

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Video Background Remover & Changer\n### You can replace image background with any color, image or video.\nNOTE: As this Space is running on ZERO GPU it has limit. It can handle approx 200frmaes at once. So, if you have big video than use small chunks or Duplicate this space.")
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
        fast_mode_checkbox = gr.Checkbox(label="Fast Mode (Use BiRefNet_lite)", value=True, interactive=True)

    time_textbox = gr.Textbox(label="Time Elapsed", interactive=False) # Add time textbox

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
        outputs=[stream_image, out_video, time_textbox],
        fn=fn,
        cache_examples=True,
        cache_mode="eager",
    )


    submit_button.click(
        fn,
        inputs=[in_video, bg_type, bg_image, bg_video, color_picker, fps_slider, video_handling_radio, fast_mode_checkbox],
        outputs=[stream_image, out_video, time_textbox],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)