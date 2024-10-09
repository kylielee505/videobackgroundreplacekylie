import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

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

def fn(vid):
    # TODO 
    # loop over video and extract images and process each one
    im = load_img(vid, output_type="pil")
    im = im.convert("RGB")
    image = process(im)
    return image

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image
  
def process_file(f):
    name_path = f.rsplit(".",1)[0]+".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

in_video = gr.Video(label="birefnet")
out_video = gr.Video()


url = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"
demo = gr.Interface(
    fn, inputs=in_video, outputs=out_video, api_name="image"
)


if __name__ == "__main__":
    demo.launch(show_error=True)