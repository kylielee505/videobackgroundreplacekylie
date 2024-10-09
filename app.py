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
def fn(vid):
    # Load the video using moviepy
    video = mp.VideoFileClip(vid)

    # Extract audio from the video
    audio = video.audio

    # Extract frames at 12 fps
    frames = video.iter_frames(fps=12)

    # Process each frame for background removal
    processed_frames = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        processed_image = process(pil_image)
        processed_frames.append(np.array(processed_image))

    # Create a new video from the processed frames
    processed_video = mp.ImageSequenceClip(processed_frames, fps=12)

    # Add the original audio back to the processed video
    processed_video = processed_video.set_audio(audio)

    # Return the processed video
    return processed_video



def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    # Create a green screen image
    green_screen = Image.new("RGBA", image_size, (0, 255, 0, 255))

    # Composite the image onto the green screen using the mask
    image = Image.composite(image, green_screen, mask)

    return image


def process_file(f):
    name_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path


in_video = gr.Video(label="birefnet")
out_video = gr.Video()


demo = gr.Interface(
    fn, inputs=in_video, outputs=out_video, api_name="video"
)


if __name__ == "__main__":
    demo.launch(show_error=True)