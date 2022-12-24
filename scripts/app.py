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
from PIL import Image as im


providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

def get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

def rmbg_fn(img):
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    return mask, img

class Script(scripts.Script):
    is_txt2img = False

    def title(self):
        return "ABG Remover"

    def show(self, is_img2img):
        return True

    def run(self, p):
        proc = process_images(p) 
        for i in range(len(proc.images)):
            nmask,nimg = rmbg_fn(np.array(proc.images[i]))
            img = im.fromarray(nimg)
            mask = im.fromarray(nmask)
            
            images.save_image(mask, p.outpath_samples, "mask_", proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)
            images.save_image(img, p.outpath_samples, "img_", proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)
            proc.images.append(mask)
            proc.images.append(img)
        return proc