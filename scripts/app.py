import random
import modules.scripts as scripts
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import gradio as gr
import huggingface_hub
import onnxruntime as rt
import copy
import numpy as np
import cv2
from PIL import Image as im, ImageDraw


# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

# Function to get mask
def get_mask(img, s=1024):
    # Resize the img to a square shape with dimension s
    # Convert img pixel values from integers 0-255 to float 0-1
    img = (img / 255).astype(np.float32)
    # get the amount of dimensions of img
    dim = img.shape[2]
    # Convert the input image to RGB format if it has an alpha channel
    if dim == 4:
        img = img[..., :3]
        dim = 3
    # Get height and width of the image
    h, w = h0, w0 = img.shape[:-1]
    # IF height is greater than width, set h as s and w as s*width/height
    # ELSE, set w as s and h as s*height/width
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    # Calculate padding for height and width
    ph, pw = s - h, s - w
    # Create a 1024x1024x3 array of 0's
    img_input = np.zeros([s, s, dim], dtype=np.float32)
    # Resize the original image to (w,h) and then pad with the calculated ph,pw
    img_input[ph // 2:ph // 2 + h, pw //
              2:pw // 2 + w] = cv2.resize(img, (w, h))
    # Change the axes
    img_input = np.transpose(img_input, (2, 0, 1))
    # Add an extra axis (1,0)
    img_input = img_input[np.newaxis, :]
    # Run the model to get the mask
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    # Transpose axis
    mask = np.transpose(mask, (1, 2, 0))
    # Crop it to the images original dimensions (h0,w0)
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    # Resize the mask to original image size (h0,w0)
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

### Function to remove background
def rmbg_fn(img):
    # Call get_mask() to get the mask
    mask = get_mask(img)
    # Multiply the image and the mask together to get the output image
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    # Convert mask value back to int 0-255
    mask = (mask * 255).astype(np.uint8)
    # Concatenate the output image and mask
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    # Stacking 3 identical copies of the mask for displaying
    mask = mask.repeat(3, axis=2)
    return mask, img


class Script(scripts.Script):
    is_txt2img = False

    # Function to set title
    def title(self):
        return "ABG Remover"

    def ui(self, is_img2img):

        with gr.Column():
            only_save_background_free_pictures = gr.Checkbox(
                label='Only save background free pictures')
            do_not_auto_save = gr.Checkbox(label='Do not auto save')
            with gr.Row():
                custom_background = gr.Checkbox(label='Custom Background')
                custom_background_color = gr.ColorPicker(
                    label='Background Color', default='#ff0000')
            custom_background_random = gr.Checkbox(
                label='Random Custom Background')

        return [only_save_background_free_pictures, do_not_auto_save, custom_background, custom_background_color, custom_background_random]

    # Function to show the script
    def show(self, is_img2img):
        return True

    # Function to run the script
    def run(self, p, only_save_background_free_pictures, do_not_auto_save, custom_background, custom_background_color, custom_background_random):
        # If only_save_background_free_pictures is true, set do_not_save_samples to true
        if only_save_background_free_pictures:
            p.do_not_save_samples = True

        # Create a process_images object
        proc = process_images(p)

        has_grid = False

        unwanted_grid_because_of_img_count = len(
            proc.images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            has_grid = True

        # Loop through all the images in proc
        for i in range(len(proc.images)):
            # Separate the background from the foreground
            nmask, nimg = rmbg_fn(np.array(proc.images[i]))

            # Convert the image back to a format that can be saved
            img = im.fromarray(nimg)

            # If only_save_background_free_pictures is true, check if the image has a background
            if custom_background or custom_background_random:
                # If custom_background_random is true, set the background color to a random color
                if custom_background_random:
                    custom_background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Create a new image with the same size as the original image
                background = im.new('RGBA', img.size, custom_background_color)

                # Draw a colored rectangle onto the new image
                draw = ImageDraw.Draw(background)
                draw.rectangle([(0, 0), img.size],
                               fill=custom_background_color)

                # Merge the two images
                img = im.alpha_composite(background, img)

            # determine output path
            outpath = p.outpath_grids if has_grid and i == 0 else p.outpath_samples

            # If we are saving all images, save the mask and the image
            if not only_save_background_free_pictures:
                mask = im.fromarray(nmask)
                # Dot not save the new images if checkbox is checked
                if not do_not_auto_save:
                    # Save the new images
                    images.save_image(
                        mask, outpath, "mask_", proc.seed + i, proc.prompt, "png", info=proc.info, p=p)
                    images.save_image(
                        img, outpath, "img_", proc.seed + i, proc.prompt, "png", info=proc.info, p=p)
                # Add the images to the proc object
                proc.images.append(mask)
                proc.images.append(img)
            # If we are only saving background-free images, save the image and replace it in the proc object
            else:
                proc.images[i] = img

            # Check if automatic saving is enabled
            if not do_not_auto_save:
                # Check if the image is the first one and has a grid
                if has_grid and i == 0:
                    # Save the image
                    images.save_image(img, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0],
                                      opts.grid_format, info=proc.info, short_filename=not opts.grid_extended_filename, p=p)
                else:
                    # Save the image
                    images.save_image(img, outpath, "", proc.seed,
                                      proc.prompt, "png", info=proc.info, p=p)

        # Return the proc object
        return proc
