import gradio as gr
import huggingface_hub
import onnxruntime as rt
import numpy as np
import cv2
from modules import script_callbacks
`
def on_ui_tabs():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
    rmbg_model = rt.InferenceSession(model_path, providers=providers)

    with gr.Blocks(analytics_enabled=False) as background_remover:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="input image")
                examples_data = [[f"examples/{x:02d}.jpg"] for x in range(1, 4)]
                examples = gr.Dataset(components=[input_img], samples=examples_data)
            run_btn = gr.Button(variant="primary")
            output_mask = gr.Image(label="mask")
            output_img = gr.Image(label="result", image_mode="RGBA")
        examples.click(lambda x: x[0], [examples], [input_img])
        run_btn.click(rmbg_fn, [input_img], [output_mask, output_img])

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
    return (background_remover, "Background Remover", "background_remover"),

script_callbacks.on_ui_tabs(on_ui_tabs)