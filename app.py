import os
import random

import cv2
import gradio as gr
from gradio_client import Client

model = os.path.join(os.path.dirname(__file__), "models/yifeng_online/Yifeng_1.png")

MODEL_MAP = {
    "Yifeng_0": 'models/yifeng_online/Yifeng_0.png',
    "Yifeng_1": 'models/yifeng_online/Yifeng_1.png',
    "Yifeng_2": 'models/yifeng_online/Yifeng_2.png',
    "Yifeng_3": 'models/yifeng_online/Yifeng_3.png',
    "Rouyan_0": 'models/rouyan_new/Rouyan_0.png',
    "Rouyan_1": 'models/rouyan_new/Rouyan_1.png',
    "Rouyan_2": 'models/rouyan_new/Rouyan_2.png',
    "Eva_0": 'models/eva/Eva_0.png',
    "Eva_1": 'models/eva/Eva_1.png',
    "Simon_0": 'models/simon_online/Simon_0.png',
    "Simon_1": 'models/simon_online/Simon_1.png',
    "Xuanxuan_0": 'models/xiaoxuan_online/Xuanxuan_0.png',
    "Xuanxuan_1": 'models/xiaoxuan_online/Xuanxuan_1.png',
    "Xuanxuan_2": 'models/xiaoxuan_online/Xuanxuan_2.png',
    "Yaqi_0": 'models/yaqi/Yaqi_0.png',
    "Yaqi_1": 'models/yaqi/Yaqi_1.png',
}


def get_tryon_result(model_name, garment1, garment2):
    model_path = MODEL_MAP[model_name]

    client = Client('https://humanaigc-outfitanyone.hf.space/--replicas/o90fr/')
    seed = random.randint(0, 1222222222)
    result = client.predict(
        model_path,
        garment1,
        garment2,
        api_name="/get_tryon_result",
        fn_index=seed
    )
    print(result)

    return remove_watermark(result)


def remove_watermark(path):
    img = cv2.imread(path)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img_without_watermark = img_[:h - 50, :, :]

    return img_without_watermark


with gr.Blocks(css=".output-image, .input-image, .image-preview {height: 400px !important} ") as demo:
    gr.HTML(
        """
        <div style="text-align: center;">
            <h1 >AI 虚拟换装</h1>
        </div>
        """)
    with gr.Row():
        with gr.Column():
            init_image = gr.Image(type="numpy", label="", value=model)
            name = gr.Label(value="Yifeng_1", label="模特名称", visible=False)
            example = gr.Examples(inputs=[name, init_image],
                                  label="模特列表",
                                  examples_per_page=3,
                                  examples=[[n, os.path.join(os.path.dirname(__file__), MODEL_MAP[n])] for n in
                                            MODEL_MAP.keys()],
                                  elem_id='example_table'
                                  )
        with gr.Column():
            with gr.Row():
                garment_top = gr.Image(sources=['upload'], type="filepath", label="上衣")
                example_top = gr.Examples(inputs=garment_top,
                                          label="样例",
                                          examples_per_page=5,
                                          examples=[
                                              os.path.join(os.path.dirname(__file__), "garments/top222.JPG"),
                                              os.path.join(os.path.dirname(__file__), "garments/top5.png"),
                                              os.path.join(os.path.dirname(__file__), "garments/top333.png"),
                                              os.path.join(os.path.dirname(__file__), "garments/dress1.png"),
                                              os.path.join(os.path.dirname(__file__), "garments/dress2.png"),
                                          ])
                garment_down = gr.Image(sources=['upload'], type="filepath", label="下衣")
                example_down = gr.Examples(inputs=garment_down, examples_per_page=5, label="样例",
                                           examples=[os.path.join(os.path.dirname(__file__), "garments/bottom1.png"),
                                                      os.path.join(os.path.dirname(__file__), "garments/bottom2.PNG"),
                                                      os.path.join(os.path.dirname(__file__), "garments/bottom3.JPG"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom4.PNG"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom5.png"),
                                                     ])

            run_button = gr.Button(value="开始换装")
        with gr.Column():
            gallery = gr.Image()
            gallery.label = "换装结果"
            run_button.click(fn=get_tryon_result,
                             inputs=[
                                 name,
                                 garment_top,
                                 garment_down,
                             ],
                             outputs=[gallery], )

if __name__ == "__main__":
    demo.title = "AI虚拟换装"
    demo.queue(max_size=10)
    demo.launch(server_name="0.0.0.0", share=True)