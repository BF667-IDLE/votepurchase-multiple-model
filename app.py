import spaces
import gradio as gr
import numpy as np
import os

# DiffuseCraft
from dc import (infer, _infer, pass_result, get_diffusers_model_list, get_samplers, save_image_history,
    get_vaes, enable_diffusers_model_detail, extract_exif_data, process_upscale, UPSCALER_KEYS, FACE_RESTORATION_MODELS,
    preset_quality, preset_styles, process_style_prompt, get_all_lora_tupled_list, update_loras, apply_lora_prompt,
    download_my_lora, search_civitai_lora, update_civitai_selection, select_civitai_lora, search_civitai_lora_json,
    get_t2i_model_info, get_civitai_tag, CIVITAI_SORT, CIVITAI_PERIOD, CIVITAI_BASEMODEL,
    SCHEDULE_TYPE_OPTIONS, SCHEDULE_PREDICTION_TYPE_OPTIONS, preprocessor_tab, SDXL_TASK, TASK_MODEL_LIST,
    PROMPT_W_OPTIONS, POST_PROCESSING_SAMPLER, DIFFUSERS_CONTROLNET_MODEL, IP_MODELS, MODE_IP_OPTIONS,
    TASK_AND_PREPROCESSORS, update_task_options, change_preprocessor_choices, get_ti_choices,
    update_textual_inversion, set_textual_inversion_prompt, create_mask_now)
# Tagger
from tagger.v2 import v2_upsampling_prompt, V2_ALL_MODELS
from tagger.utils import (gradio_copy_text, gradio_copy_prompt, COPY_ACTION_JS,
    V2_ASPECT_RATIO_OPTIONS, V2_RATING_OPTIONS, V2_LENGTH_OPTIONS, V2_IDENTITY_OPTIONS)
from tagger.tagger import (predict_tags_wd, convert_danbooru_to_e621_prompt,
    remove_specific_prompt, insert_recom_prompt, compose_prompt_to_copy,
    translate_prompt, select_random_character)
from tagger.fl2sd3longcap import predict_tags_fl2_sd3

def description_ui():
    gr.Markdown(
        """
## Danbooru Tags Transformer V2 Demo with WD Tagger & SD3 Long Captioner
(Image =>) Prompt => Upsampled longer prompt
- Mod of p1atdev's [Danbooru Tags Transformer V2 Demo](https://huggingface.co/spaces/p1atdev/danbooru-tags-transformer-v2) and [WD Tagger with 🤗 transformers](https://huggingface.co/spaces/p1atdev/wd-tagger-transformers).
- Models: p1atdev's [wd-swinv2-tagger-v3-hf](https://huggingface.co/p1atdev/wd-swinv2-tagger-v3-hf), [dart-v2-moe-sft](https://huggingface.co/p1atdev/dart-v2-moe-sft), [dart-v2-sft](https://huggingface.co/p1atdev/dart-v2-sft)\
, gokaygokay's [Florence-2-SD3-Captioner](https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner)
"""
    )

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 4096
MIN_IMAGE_SIZE = 256

css = """
#container { margin: 0 auto; !important; }
#col-container { margin: 0 auto; !important; }
#result { max-width: 520px; max-height: 520px; width: 520px; height: 520px; margin: 0px auto; object-fit: contain; !important; }
.lora { min-width: 480px; !important; }
.title { font-size: 3em; align-items: center; text-align: center; }
.info { align-items: center; text-align: center; }
.desc [src$='#float'] { float: right; margin: 20px; }
.image { margin: 0px auto; object-fit: contain; }
"""

with gr.Blocks(fill_width=True, elem_id="container", css=css) as demo:
    gr.Markdown("# Votepurchase Multiple Model", elem_classes="title")
    state = gr.State(value={})
    with gr.Tab("Image Generator"):
        with gr.Column(elem_id="col-container"):
            with gr.Row():
                prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=8, placeholder="Enter your prompt", container=False)
            
            with gr.Row():
                run_button = gr.Button("Run", variant="primary", scale=5)
                auto_trans = gr.Checkbox(label="Auto translate to English", value=False, scale=2)

            result = gr.Image(label="Result", elem_id="result", format="png", type="filepath", show_label=False, interactive=False,
                              show_download_button=True, show_share_button=False, container=True)

            with gr.Accordion("History", open=False):
                history_files = gr.Files(interactive=False, visible=False)
                history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", format="png", interactive=False, show_share_button=False,
                show_download_button=True)
                history_clear_button = gr.Button(value="Clear History", variant="secondary")
                history_clear_button.click(lambda: ([], []), None, [history_gallery, history_files], queue=False, show_api=False)

            with gr.Accordion("Advanced Settings", open=True):
                task = gr.Dropdown(label="Task", choices=SDXL_TASK, value=TASK_MODEL_LIST[0])
                with gr.Tab("Generation Settings"):
                    with gr.Row():
                        negative_prompt = gr.Text(label="Negative prompt", lines=1, max_lines=6, placeholder="Enter a negative prompt", show_copy_button=True,
                                                value="(low quality, worst quality:1.2), very displeasing, watermark, signature, ugly")
                    with gr.Accordion("Prompt Settings", open=False):
                        with gr.Row():
                            quality_selector = gr.Radio(label="Quality Tag Presets", interactive=True, choices=list(preset_quality.keys()), value="None", scale=3)
                            style_selector = gr.Radio(label="Style Presets", interactive=True, choices=list(preset_styles.keys()), value="None", scale=3)
                        with gr.Row():
                            recom_prompt = gr.Checkbox(label="Recommended prompt", value=True, scale=1)
                            prompt_syntax = gr.Dropdown(label="Prompt Syntax", choices=PROMPT_W_OPTIONS, value=PROMPT_W_OPTIONS[1][1])
                    with gr.Row():
                        with gr.Column(scale=4):
                            model_name = gr.Dropdown(label="Model", info="You can enter a huggingface model repo_id to want to use.",
                                                    choices=get_diffusers_model_list(), value=get_diffusers_model_list()[0],
                                                    allow_custom_value=True, interactive=True, min_width=320)
                            model_info = gr.Markdown(elem_classes="info")
                        with gr.Column(scale=1):
                            model_detail = gr.Checkbox(label="Show detail of model in list", value=False)
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        gpu_duration = gr.Slider(label="GPU time duration (seconds)", minimum=5, maximum=240, value=20)
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=MIN_IMAGE_SIZE, maximum=MAX_IMAGE_SIZE, step=32, value=1024) # 832
                        height = gr.Slider(label="Height", minimum=MIN_IMAGE_SIZE, maximum=MAX_IMAGE_SIZE, step=32, value=1024) # 1216
                        guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=30.0, step=0.1, value=7)
                        guidance_rescale = gr.Slider(label="CFG rescale", value=0., step=0.01, minimum=0., maximum=1.5)
                    with gr.Row():
                        num_inference_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=100, step=1, value=28)
                        pag_scale = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="PAG Scale")
                        clip_skip = gr.Checkbox(value=True, label="Layer 2 Clip Skip")
                        free_u = gr.Checkbox(value=False, label="FreeU")
                    with gr.Row():
                        sampler = gr.Dropdown(label="Sampler", choices=get_samplers(), value="Euler")
                        schedule_type = gr.Dropdown(label="Schedule type", choices=SCHEDULE_TYPE_OPTIONS, value=SCHEDULE_TYPE_OPTIONS[0])
                        schedule_prediction_type = gr.Dropdown(label="Discrete Sampling Type", choices=SCHEDULE_PREDICTION_TYPE_OPTIONS, value=SCHEDULE_PREDICTION_TYPE_OPTIONS[0])
                        vae_model = gr.Dropdown(label="VAE Model", choices=get_vaes(), value=get_vaes()[0])
                    with gr.Accordion("Other Settings", open=False):
                        with gr.Accordion("Textual inversion", open=True):
                            active_textual_inversion = gr.Checkbox(value=False, label="Active Textual Inversion in prompt")
                            use_textual_inversion = gr.CheckboxGroup(choices=get_ti_choices(model_name.value) if active_textual_inversion.value else [], value=None, label="Use Textual Invertion in prompt")

                with gr.Tab("LoRA"):
                    def lora_dropdown(label, visible=True):
                        return gr.Dropdown(label=label, choices=get_all_lora_tupled_list(), value="", allow_custom_value=True, elem_classes="lora", min_width=320, visible=visible)

                    def lora_scale_slider(label, visible=True):
                        val_lora = 8 if IS_ZERO_GPU else 8 #
                        return gr.Slider(minimum=-val_lora, maximum=val_lora, step=0.01, value=1.0, label=label, visible=visible)
                    
                    def lora_textbox():
                        return gr.Textbox(label="", info="Example of prompt:", value="", show_copy_button=True, interactive=False, visible=False)
                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                lora1 = lora_dropdown("LoRA 1")
                                lora1_wt = lora_scale_slider("LoRA 1: weight")
                            with gr.Row():
                                lora1_info = lora_textbox()
                                lora1_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora1_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora2 = lora_dropdown("LoRA 2")
                                lora2_wt = lora_scale_slider("LoRA 2: weight")
                            with gr.Row():
                                lora2_info = lora_textbox()
                                lora2_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora2_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora3 = lora_dropdown("LoRA 3")
                                lora3_wt = lora_scale_slider("LoRA 3: weight")
                            with gr.Row():
                                lora3_info = lora_textbox()
                                lora3_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora3_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora4 = lora_dropdown("LoRA 4")
                                lora4_wt = lora_scale_slider("LoRA 4: weight")
                            with gr.Row():
                                lora4_info = lora_textbox()
                                lora4_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora4_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora5 = lora_dropdown("LoRA 5")
                                lora5_wt = lora_scale_slider("LoRA 5: weight")
                            with gr.Row():
                                lora5_info = lora_textbox()
                                lora5_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora5_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora6 = lora_dropdown("LoRA 6")
                                lora6_wt = lora_scale_slider("LoRA 6: weight")
                            with gr.Row():
                                lora6_info = lora_textbox()
                                lora6_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora6_md = gr.Markdown(value="", visible=False)
                        with gr.Column():
                            with gr.Row():
                                lora7 = lora_dropdown("LoRA 7")
                                lora7_wt = lora_scale_slider("LoRA 7: weight")
                            with gr.Row():
                                lora7_info = lora_textbox()
                                lora7_copy = gr.Button(value="Copy example to prompt", visible=False)
                            lora7_md = gr.Markdown(value="", visible=False)
                    with gr.Accordion("From URL", open=True, visible=True):
                        with gr.Row():
                            lora_search_civitai_basemodel = gr.CheckboxGroup(label="Search LoRA for", choices=CIVITAI_BASEMODEL, value=["Pony", "Illustrious", "SDXL 1.0"])
                            lora_search_civitai_sort = gr.Radio(label="Sort", choices=CIVITAI_SORT, value="Highest Rated")
                            lora_search_civitai_period = gr.Radio(label="Period", choices=CIVITAI_PERIOD, value="AllTime")
                        with gr.Row():
                            lora_search_civitai_query = gr.Textbox(label="Query", placeholder="oomuro sakurako...", lines=1)
                            lora_search_civitai_tag = gr.Dropdown(label="Tag", choices=get_civitai_tag(), value=get_civitai_tag()[0], allow_custom_value=True)
                            lora_search_civitai_user = gr.Textbox(label="Username", lines=1)
                        lora_search_civitai_submit = gr.Button("Search on Civitai")
                        with gr.Row():
                            lora_search_civitai_json = gr.JSON(value={}, visible=False)
                            lora_search_civitai_desc = gr.Markdown(value="", visible=False, elem_classes="desc")
                        with gr.Accordion("Select from Gallery", open=False):
                            lora_search_civitai_gallery = gr.Gallery([], label="Results", allow_preview=False, columns=5, show_share_button=False, interactive=False)
                        lora_search_civitai_result = gr.Dropdown(label="Search Results", choices=[("", "")], value="", allow_custom_value=True, visible=False)
                        lora_download_url = gr.Textbox(label="LoRA's download URL", placeholder="https://civitai.com/api/download/models/28907", info="It has to be .safetensors files, and you can also download them from Hugging Face.", lines=1)
                        lora_download = gr.Button("Get and set LoRA and apply to prompt")
                
                with gr.Tab("ControlNet / Img2img / Inpaint"):
                    task_sel = gr.Radio(label="Task Selector", choices=SDXL_TASK, value=TASK_MODEL_LIST[0])
                    with gr.Row():
                        with gr.Column():
                            #image_control = gr.Image(label="Image ControlNet / Inpaint / Img2img", type="filepath", height=384, sources=["upload", "clipboard", "webcam"], show_share_button=False)
                            image_control = gr.ImageEditor(label="Image ControlNet / Inpaint / Img2img", type="filepath", sources=["upload", "clipboard", "webcam"], image_mode='RGB',
                                                        show_share_button=False, show_fullscreen_button=False, layers=False, canvas_size=(384, 384), width=384, height=512,
                                                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=32), eraser=gr.Eraser(default_size="32"), elem_classes="image")
                            result_to_ic_button = gr.Button("Get image from generated result")
                        image_mask = gr.Image(label="Image Mask", type="filepath", height=384, sources=["upload", "clipboard"], show_share_button=False, elem_classes="image")
                    with gr.Row():
                        strength = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.55, label="Strength",
                                            info="This option adjusts the level of changes for img2img, repaint and inpaint.")
                        image_resolution = gr.Slider(minimum=64, maximum=2048, step=64, value=1024, label="Image Resolution",
                                                    info="The maximum proportional size of the generated image based on the uploaded image.")
                    with gr.Row():
                        controlnet_model = gr.Dropdown(label="ControlNet model", choices=DIFFUSERS_CONTROLNET_MODEL, value=DIFFUSERS_CONTROLNET_MODEL[0], allow_custom_value=True)
                    with gr.Row():
                        control_net_output_scaling = gr.Slider(minimum=0, maximum=5.0, step=0.1, value=1, label="ControlNet Output Scaling in UNet")
                        control_net_start_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="ControlNet Start Threshold (%)")
                        control_net_stop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="ControlNet Stop Threshold (%)")
                    with gr.Row():
                        preprocessor_name = gr.Dropdown(label="Preprocessor Name", choices=TASK_AND_PREPROCESSORS["canny"])
                    with gr.Row():
                        preprocess_resolution = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Preprocessor Resolution")
                        low_threshold = gr.Slider(minimum=1, maximum=255, step=1, value=100, label="'CANNY' low threshold")
                        high_threshold = gr.Slider(minimum=1, maximum=255, step=1, value=200, label="'CANNY' high threshold")
                    with gr.Row():
                        value_threshold = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.1, label="'MLSD' Hough value threshold")
                        distance_threshold = gr.Slider(minimum=0.0, maximum=20.0, step=0.01, value=0.1, label="'MLSD' Hough distance threshold")
                    recolor_gamma_correction = gr.Number(minimum=0., maximum=25., value=1., step=0.001, label="'RECOLOR' gamma correction")
                    tile_blur_sigma = gr.Number(minimum=0, maximum=100, value=9, step=1, label="'TILE' blur sigma")

                with gr.Tab("IP-Adapter"):
                    with gr.Accordion("IP-Adapter 1", open=True, visible=True):
                        with gr.Row():
                            with gr.Column():
                                #image_ip1 = gr.Image(label="IP Image", type="filepath", height=384, sources=["upload", "clipboard"], show_share_button=False)
                                image_ip1 = gr.ImageEditor(label="IP Image", type="filepath", sources=["upload", "clipboard", "webcam"], image_mode='RGB',
                                                        show_share_button=False, show_fullscreen_button=False, layers=False, canvas_size=(384, 384), width=384, height=512,
                                                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=32), eraser=gr.Eraser(default_size="32"), elem_classes="image")
                                result_to_ip1_button = gr.Button("Get image from generated result")
                            mask_ip1 = gr.Image(label="IP Mask (optional)", type="filepath", height=384, sources=["upload", "clipboard"], show_share_button=False, elem_classes="image")
                        with gr.Row():
                            model_ip1 = gr.Dropdown(value="plus_face", label="Model", choices=IP_MODELS)
                            mode_ip1 = gr.Dropdown(value="original", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip1 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")
                    with gr.Accordion("IP-Adapter 2", open=True, visible=True):
                        with gr.Row():
                            with gr.Column():
                                #image_ip2 = gr.Image(label="IP Image", type="filepath", height=384, sources=["upload", "clipboard"], show_share_button=False)
                                image_ip2 = gr.ImageEditor(label="IP Image", type="filepath", sources=["upload", "clipboard", "webcam"], image_mode='RGB',
                                                        show_share_button=False, show_fullscreen_button=False, layers=False, canvas_size=(384, 384), width=384, height=512,
                                                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=32), eraser=gr.Eraser(default_size="32"), elem_classes="image")
                                result_to_ip2_button = gr.Button("Get image from generated result")
                            mask_ip2 = gr.Image(label="IP Mask (optional)", type="filepath", height=384, sources=["upload", "clipboard"], show_share_button=False, elem_classes="image")
                        with gr.Row():
                            model_ip2 = gr.Dropdown(value="base", label="Model", choices=IP_MODELS)
                            mode_ip2 = gr.Dropdown(value="style", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip2 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")

                with gr.Tab("Inpaint Mask Maker"):
                    with gr.Row():
                        with gr.Column():
                            image_base = gr.ImageEditor(sources=["upload", "clipboard", "webcam"],
                                brush=gr.Brush(default_size="32", color_mode="fixed", colors=["rgba(0, 0, 0, 1)", "rgba(0, 0, 0, 0.1)", "rgba(255, 255, 255, 0.1)"]),
                                eraser=gr.Eraser(default_size="32"), show_share_button=False, show_fullscreen_button=False,
                                canvas_size=(384, 384), width=384, height=512, elem_classes="image")
                            result_to_cm_button = gr.Button("Get image from generated result")
                            invert_mask = gr.Checkbox(value=False, label="Invert mask")
                            cm_btn = gr.Button("Create mask")
                        with gr.Column():
                            img_source = gr.Image(interactive=False, height=384, show_share_button=False, elem_classes="image")
                            img_result = gr.Image(label="Mask image", show_label=True, interactive=False, height=384, show_share_button=False, elem_classes="image")
                            cm_btn_send = gr.Button("Send to ControlNet / Img2img / Inpaint")
                            with gr.Row():
                                cm_btn_send_ip1 = gr.Button("Send to IP-Adapter 1")
                                cm_btn_send_ip2 = gr.Button("Send to IP-Adapter 2")
                        cm_btn.click(create_mask_now, [image_base, invert_mask], [img_source, img_result], show_api=False)
                        def send_img(img_source, img_result):
                            return img_source, img_result
                        cm_btn_send.click(send_img, [img_source, img_result], [image_control, image_mask], queue=False, show_api=False)
                        cm_btn_send_ip1.click(send_img, [img_source, img_result], [image_ip1, mask_ip1], queue=False, show_api=False)
                        cm_btn_send_ip2.click(send_img, [img_source, img_result], [image_ip2, mask_ip2], queue=False, show_api=False)

                with gr.Tab("Hires fix / Detailfix / Face restoration"):
                    with gr.Accordion("Hires fix", open=True):
                        with gr.Row():
                            upscaler_model_path = gr.Dropdown(label="Upscaler", choices=UPSCALER_KEYS, value=UPSCALER_KEYS[0])
                            upscaler_increases_size = gr.Slider(minimum=1.1, maximum=4., step=0.1, value=1.2, label="Upscale by")
                            upscaler_tile_size = gr.Slider(minimum=0, maximum=512, step=16, value=(0 if IS_ZERO_GPU else 192), label="Upscaler Tile Size", info="0 = no tiling")
                            upscaler_tile_overlap = gr.Slider(minimum=0, maximum=48, step=1, value=8, label="Upscaler Tile Overlap")
                        with gr.Row():
                            hires_steps = gr.Slider(minimum=0, value=30, maximum=100, step=1, label="Hires Steps")
                            hires_denoising_strength = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.55, label="Hires Denoising Strength")
                            hires_sampler = gr.Dropdown(label="Hires Sampler", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])
                            hires_schedule_list = ["Use same schedule type"] + SCHEDULE_TYPE_OPTIONS
                            hires_schedule_type = gr.Dropdown(label="Hires Schedule type", choices=hires_schedule_list, value=hires_schedule_list[0])
                            hires_guidance_scale = gr.Slider(minimum=-1., maximum=30., step=0.5, value=-1., label="Hires CFG", info="If the value is -1, the main CFG will be used")
                        with gr.Row():
                            hires_prompt = gr.Textbox(label="Hires Prompt", placeholder="Main prompt will be use", lines=3)
                            hires_negative_prompt = gr.Textbox(label="Hires Negative Prompt", placeholder="Main negative prompt will be use", lines=3)
                    with gr.Accordion("Detail fix", open=True):
                        with gr.Row():
                            # Adetailer Inpaint Only
                            adetailer_inpaint_only = gr.Checkbox(label="Inpaint only", value=True)
                            # Adetailer Verbose
                            adetailer_verbose = gr.Checkbox(label="Verbose", value=False)
                            # Adetailer Sampler
                            adetailer_sampler = gr.Dropdown(label="Adetailer sampler:", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])
                        with gr.Accordion("Detailfix A", open=True, visible=True):
                            # Adetailer A
                            adetailer_active_a = gr.Checkbox(label="Enable Adetailer A", value=False)
                            prompt_ad_a = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                            negative_prompt_ad_a = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                            with gr.Row():
                                strength_ad_a = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                                face_detector_ad_a = gr.Checkbox(label="Face detector", value=False)
                                person_detector_ad_a = gr.Checkbox(label="Person detector", value=True)
                                hand_detector_ad_a = gr.Checkbox(label="Hand detector", value=False)
                            with gr.Row():
                                mask_dilation_a = gr.Number(label="Mask dilation:", value=4, minimum=1)
                                mask_blur_a = gr.Number(label="Mask blur:", value=4, minimum=1)
                                mask_padding_a = gr.Number(label="Mask padding:", value=32, minimum=1)
                        with gr.Accordion("Detailfix B", open=True, visible=True):
                            # Adetailer B
                            adetailer_active_b = gr.Checkbox(label="Enable Adetailer B", value=False)
                            prompt_ad_b = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                            negative_prompt_ad_b = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                            with gr.Row():
                                strength_ad_b = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                                face_detector_ad_b = gr.Checkbox(label="Face detector", value=False)
                                person_detector_ad_b = gr.Checkbox(label="Person detector", value=True)
                                hand_detector_ad_b = gr.Checkbox(label="Hand detector", value=False)
                            with gr.Row():
                                mask_dilation_b = gr.Number(label="Mask dilation:", value=4, minimum=1)
                                mask_blur_b = gr.Number(label="Mask blur:", value=4, minimum=1)
                                mask_padding_b = gr.Number(label="Mask padding:", value=32, minimum=1)
                    with gr.Accordion("Face restoration", open=True, visible=True):
                        face_rest_options = [None] + FACE_RESTORATION_MODELS
                        with gr.Row():
                            face_restoration_model = gr.Dropdown(label="Face restoration model", choices=face_rest_options, value=face_rest_options[0])
                            face_restoration_visibility = gr.Slider(minimum=0., maximum=1., step=0.001, value=1., label="Visibility")
                            face_restoration_weight = gr.Slider(minimum=0., maximum=1., step=0.001, value=.5, label="Weight", info="(0 = maximum effect, 1 = minimum effect)")

        examples = gr.Examples(
            examples = [
                ["souryuu asuka langley, 1girl, neon genesis evangelion, plugsuit, pilot suit, red bodysuit, sitting, crossing legs, black eye patch, cat hat, throne, symmetrical, looking down, from bottom, looking at viewer, outdoors"],
                ["sailor moon, magical girl transformation, sparkles and ribbons, soft pastel colors, crescent moon motif, starry night sky background, shoujo manga style"],
                ["kafuu chino, 1girl, solo"],
                ["1girl"],
                ["beautiful sunset"],
            ],
            inputs=[prompt],
            cache_examples=False,
        )

    model_name.change(update_task_options, [model_name, task], [task], queue=False, show_api=False)\
    .success(update_task_options, [model_name, task_sel], [task_sel], queue=False, show_api=False)\
    .success(update_textual_inversion, [active_textual_inversion, model_name], [use_textual_inversion], queue=False, show_api=False)
    task_sel.select(lambda x: x, [task_sel], [task], queue=False, show_api=False)
    task.change(change_preprocessor_choices, [task], [preprocessor_name], queue=False, show_api=False)\
    .success(lambda x: x, [task], [task_sel], queue=False, show_api=False)
    active_textual_inversion.change(update_textual_inversion, [active_textual_inversion, model_name], [use_textual_inversion], queue=False, show_api=False)
    use_textual_inversion.change(set_textual_inversion_prompt, [use_textual_inversion, prompt, negative_prompt, prompt_syntax], [prompt, negative_prompt])
    result_to_cm_button.click(lambda x: x, [result], [image_base], queue=False, show_api=False)
    result_to_ic_button.click(lambda x: x, [result], [image_control], queue=False, show_api=False)
    result_to_ip1_button.click(lambda x: x, [result], [image_ip1], queue=False, show_api=False)
    result_to_ip2_button.click(lambda x: x, [result], [image_ip2], queue=False, show_api=False)

    gr.on( #lambda x: None, inputs=None, outputs=result).then(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, negative_prompt, seed, randomize_seed, width, height,
                guidance_scale, num_inference_steps, model_name,
                lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt,
                lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt, task, prompt_syntax,
                sampler, vae_model, schedule_type, schedule_prediction_type,
                clip_skip, pag_scale, free_u, guidance_rescale,
                image_control, image_mask, strength, image_resolution,
                controlnet_model, control_net_output_scaling, control_net_start_threshold, control_net_stop_threshold,
                preprocessor_name, preprocess_resolution, low_threshold, high_threshold,
                value_threshold, distance_threshold, recolor_gamma_correction, tile_blur_sigma,
                image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1,
                image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2,
                upscaler_model_path, upscaler_increases_size, upscaler_tile_size, upscaler_tile_overlap, hires_steps, hires_denoising_strength,
                hires_sampler, hires_schedule_type, hires_guidance_scale, hires_prompt, hires_negative_prompt,
                adetailer_inpaint_only, adetailer_verbose, adetailer_sampler, adetailer_active_a,
                prompt_ad_a, negative_prompt_ad_a, strength_ad_a, face_detector_ad_a, person_detector_ad_a, hand_detector_ad_a,
                mask_dilation_a, mask_blur_a, mask_padding_a, adetailer_active_b, prompt_ad_b, negative_prompt_ad_b, strength_ad_b,
                face_detector_ad_b, person_detector_ad_b, hand_detector_ad_b, mask_dilation_b, mask_blur_b, mask_padding_b,
                active_textual_inversion, face_restoration_model, face_restoration_visibility, face_restoration_weight, gpu_duration, auto_trans, recom_prompt],
        outputs=[result],
        queue=True,
        show_progress="full",
        show_api=True,
    )

    result.change(save_image_history, [result, history_gallery, history_files, model_name], [history_gallery, history_files], queue=False, show_api=False)

    gr.on(
        triggers=[lora1.change, lora1_wt.change, lora2.change, lora2_wt.change, lora3.change, lora3_wt.change,
                  lora4.change, lora4_wt.change, lora5.change, lora5_wt.change, lora6.change, lora6_wt.change, lora7.change, lora7_wt.change, prompt_syntax.change],
        fn=update_loras,
        inputs=[prompt, prompt_syntax, lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt],
        outputs=[prompt, lora1, lora1_wt, lora1_info, lora1_copy, lora1_md,
                 lora2, lora2_wt, lora2_info, lora2_copy, lora2_md, lora3, lora3_wt, lora3_info, lora3_copy, lora3_md, 
                 lora4, lora4_wt, lora4_info, lora4_copy, lora4_md, lora5, lora5_wt, lora5_info, lora5_copy, lora5_md,
                 lora6, lora6_wt, lora6_info, lora6_copy, lora6_md, lora7, lora7_wt, lora7_info, lora7_copy, lora7_md],
        queue=False,
        trigger_mode="once",
        show_api=False,
    )
    lora1_copy.click(apply_lora_prompt, [prompt, lora1_info], [prompt], queue=False, show_api=False)
    lora2_copy.click(apply_lora_prompt, [prompt, lora2_info], [prompt], queue=False, show_api=False)
    lora3_copy.click(apply_lora_prompt, [prompt, lora3_info], [prompt], queue=False, show_api=False)
    lora4_copy.click(apply_lora_prompt, [prompt, lora4_info], [prompt], queue=False, show_api=False)
    lora5_copy.click(apply_lora_prompt, [prompt, lora5_info], [prompt], queue=False, show_api=False)
    lora6_copy.click(apply_lora_prompt, [prompt, lora6_info], [prompt], queue=False, show_api=False)
    lora7_copy.click(apply_lora_prompt, [prompt, lora7_info], [prompt], queue=False, show_api=False)

    gr.on(
        triggers=[lora_search_civitai_submit.click, lora_search_civitai_query.submit],
        fn=search_civitai_lora,
        inputs=[lora_search_civitai_query, lora_search_civitai_basemodel, lora_search_civitai_sort, lora_search_civitai_period, lora_search_civitai_tag, lora_search_civitai_user, lora_search_civitai_gallery],
        outputs=[lora_search_civitai_result, lora_search_civitai_desc, lora_search_civitai_submit, lora_search_civitai_query, lora_search_civitai_gallery],
        scroll_to_output=True,
        queue=True,
        show_api=False,
    )
    lora_search_civitai_json.change(search_civitai_lora_json, [lora_search_civitai_query, lora_search_civitai_basemodel], [lora_search_civitai_json], queue=True, show_api=True)  # fn for api
    lora_search_civitai_result.change(select_civitai_lora, [lora_search_civitai_result], [lora_download_url, lora_search_civitai_desc], scroll_to_output=True, queue=False, show_api=False)
    gr.on(
        triggers=[lora_download.click, lora_download_url.submit],
        fn=download_my_lora,
        inputs=[lora_download_url, lora1, lora2, lora3, lora4, lora5, lora6, lora7],
        outputs=[lora1, lora2, lora3, lora4, lora5, lora6, lora7],
        scroll_to_output=True,
        queue=True,
        show_api=False,
    )
    lora_search_civitai_gallery.select(update_civitai_selection, None, [lora_search_civitai_result], queue=False, show_api=False)

    #recom_prompt.change(enable_model_recom_prompt, [recom_prompt], [recom_prompt], queue=False, show_api=False)
    gr.on(
        triggers=[quality_selector.change, style_selector.change],
        fn=process_style_prompt,
        inputs=[prompt, negative_prompt, style_selector, quality_selector],
        outputs=[prompt, negative_prompt],
        queue=False,
        trigger_mode="once",
        show_api=False,
    )

    model_detail.change(enable_diffusers_model_detail, [model_detail, model_name, state], [model_detail, model_name, state], queue=False, show_api=False)
    model_name.change(get_t2i_model_info, [model_name], [model_info], queue=False, show_api=False)

    # Tagger
    with gr.Tab("Tags Transformer with Tagger"):
        with gr.Column():
                with gr.Group():
                    input_image = gr.Image(label="Input image", type="pil", sources=["upload", "clipboard"], height=256)
                    with gr.Accordion(label="Advanced options", open=False):
                        general_threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.01, interactive=True)
                        character_threshold = gr.Slider(label="Character threshold", minimum=0.0, maximum=1.0, value=0.8, step=0.01, interactive=True)
                        input_tag_type = gr.Radio(label="Convert tags to", info="danbooru for Animagine, e621 for Pony.", choices=["danbooru", "e621"], value="danbooru")
                        recom_prompt = gr.Radio(label="Insert reccomended prompt", choices=["None", "Animagine", "Pony"], value="None", interactive=True)
                    image_algorithms = gr.CheckboxGroup(["Use WD Tagger", "Use Florence-2-SD3-Long-Captioner"], label="Algorithms", value=["Use WD Tagger"])
                    keep_tags = gr.Radio(label="Remove tags leaving only the following", choices=["body", "dress", "all"], value="all")
                    generate_from_image_btn = gr.Button(value="GENERATE TAGS FROM IMAGE", size="lg", variant="primary")
                with gr.Group():
                    with gr.Row():
                        input_character = gr.Textbox(label="Character tags", placeholder="hatsune miku")
                        input_copyright = gr.Textbox(label="Copyright tags", placeholder="vocaloid")
                        random_character = gr.Button(value="Random character 🎲", size="sm")
                    input_general = gr.TextArea(label="General tags", lines=4, placeholder="1girl, ...", value="")
                    input_tags_to_copy = gr.Textbox(value="", visible=False)
                    with gr.Row():
                        copy_input_btn = gr.Button(value="Copy to clipboard", size="sm", interactive=False)
                        copy_prompt_btn_input = gr.Button(value="Copy to primary prompt", size="sm", interactive=False)
                    translate_input_prompt_button = gr.Button(value="Translate prompt to English", size="sm", variant="secondary")
                    tag_type = gr.Radio(label="Output tag conversion", info="danbooru for Animagine, e621 for Pony.", choices=["danbooru", "e621"], value="e621", visible=False)
                    input_rating = gr.Radio(label="Rating", choices=list(V2_RATING_OPTIONS), value="explicit")
                    with gr.Accordion(label="Advanced options", open=False):
                        input_aspect_ratio = gr.Radio(label="Aspect ratio", info="The aspect ratio of the image.", choices=list(V2_ASPECT_RATIO_OPTIONS), value="square")
                        input_length = gr.Radio(label="Length", info="The total length of the tags.", choices=list(V2_LENGTH_OPTIONS), value="very_long")
                        input_identity = gr.Radio(label="Keep identity", info="How strictly to keep the identity of the character or subject. If you specify the detail of subject in the prompt, you should choose `strict`. Otherwise, choose `none` or `lax`. `none` is very creative but sometimes ignores the input prompt.", choices=list(V2_IDENTITY_OPTIONS), value="lax")                    
                        input_ban_tags = gr.Textbox(label="Ban tags", info="Tags to ban from the output.", placeholder="alternate costumen, ...", value="censored")
                        model_name = gr.Dropdown(label="Model", choices=list(V2_ALL_MODELS.keys()), value=list(V2_ALL_MODELS.keys())[0])
                        dummy_np = gr.Textbox(label="Negative prompt", value="", visible=False)
                        recom_animagine = gr.Textbox(label="Animagine reccomended prompt", value="Animagine", visible=False)
                        recom_pony = gr.Textbox(label="Pony reccomended prompt", value="Pony", visible=False)
                    generate_btn = gr.Button(value="GENERATE TAGS", size="lg", variant="primary")
                with gr.Row():
                    with gr.Group():
                        output_text = gr.TextArea(label="Output tags", interactive=False, show_copy_button=True)
                        with gr.Row():
                            copy_btn = gr.Button(value="Copy to clipboard", size="sm", interactive=False)
                            copy_prompt_btn = gr.Button(value="Copy to primary prompt", size="sm", interactive=False)
                    with gr.Group():
                        output_text_pony = gr.TextArea(label="Output tags (Pony e621 style)", interactive=False, show_copy_button=True)
                        with gr.Row():
                            copy_btn_pony = gr.Button(value="Copy to clipboard", size="sm", interactive=False)
                            copy_prompt_btn_pony = gr.Button(value="Copy to primary prompt", size="sm", interactive=False)

        random_character.click(select_random_character, [input_copyright, input_character], [input_copyright, input_character], queue=False, show_api=False)

        translate_input_prompt_button.click(translate_prompt, [input_general], [input_general], queue=False, show_api=False)
        translate_input_prompt_button.click(translate_prompt, [input_character], [input_character], queue=False, show_api=False)
        translate_input_prompt_button.click(translate_prompt, [input_copyright], [input_copyright], queue=False, show_api=False)

        generate_from_image_btn.click(
            lambda: ("", "", ""), None, [input_copyright, input_character, input_general], queue=False, show_api=False,
        ).success(
            predict_tags_wd,
            [input_image, input_general, image_algorithms, general_threshold, character_threshold],
            [input_copyright, input_character, input_general, copy_input_btn],
            show_api=False,
        ).success(
            predict_tags_fl2_sd3, [input_image, input_general, image_algorithms], [input_general], show_api=False,
        ).success(
            remove_specific_prompt, [input_general, keep_tags], [input_general], queue=False, show_api=False,
        ).success(
            convert_danbooru_to_e621_prompt, [input_general, input_tag_type], [input_general], queue=False, show_api=False,
        ).success(
            insert_recom_prompt, [input_general, dummy_np, recom_prompt], [input_general, dummy_np], queue=False, show_api=False,
        ).success(lambda: gr.update(interactive=True), None, [copy_prompt_btn_input], queue=False, show_api=False)
        copy_input_btn.click(compose_prompt_to_copy, [input_character, input_copyright, input_general], [input_tags_to_copy], show_api=False)\
            .success(gradio_copy_text, [input_tags_to_copy], js=COPY_ACTION_JS, show_api=False)
        copy_prompt_btn_input.click(compose_prompt_to_copy, inputs=[input_character, input_copyright, input_general], outputs=[input_tags_to_copy], show_api=False)\
            .success(gradio_copy_prompt, inputs=[input_tags_to_copy], outputs=[prompt], show_api=False)
        
        generate_btn.click(
            v2_upsampling_prompt,
            [model_name, input_copyright, input_character, input_general,
            input_rating, input_aspect_ratio, input_length, input_identity, input_ban_tags],
            [output_text],
            show_api=False,
        ).success(
            convert_danbooru_to_e621_prompt, [output_text, tag_type], [output_text_pony], queue=False, show_api=False,
        ).success(
            insert_recom_prompt, [output_text, dummy_np, recom_animagine], [output_text, dummy_np], queue=False, show_api=False,
        ).success(
            insert_recom_prompt, [output_text_pony, dummy_np, recom_pony], [output_text_pony, dummy_np], queue=False, show_api=False,
        ).success(lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),
                  None, [copy_btn, copy_btn_pony, copy_prompt_btn, copy_prompt_btn_pony], queue=False, show_api=False)
        copy_btn.click(gradio_copy_text, [output_text], js=COPY_ACTION_JS, show_api=False)
        copy_btn_pony.click(gradio_copy_text, [output_text_pony], js=COPY_ACTION_JS, show_api=False)
        copy_prompt_btn.click(gradio_copy_prompt, inputs=[output_text], outputs=[prompt], show_api=False)
        copy_prompt_btn_pony.click(gradio_copy_prompt, inputs=[output_text_pony], outputs=[prompt], show_api=False)

    with gr.Tab("PNG Info"):
        with gr.Row():
            with gr.Column():
                image_metadata = gr.Image(label="Image with metadata", type="pil", sources=["upload"])

            with gr.Column():
                result_metadata = gr.Textbox(label="Metadata", show_label=True, show_copy_button=True, interactive=False, container=True, max_lines=99)

                image_metadata.change(
                    fn=extract_exif_data,
                    inputs=[image_metadata],
                    outputs=[result_metadata],
                )

    with gr.Tab("Upscaler"):
        with gr.Row():
            with gr.Column():
                USCALER_TAB_KEYS = [name for name in UPSCALER_KEYS[9:]]
                image_up_tab = gr.Image(label="Image", type="pil", sources=["upload"])
                upscaler_tab = gr.Dropdown(label="Upscaler", choices=USCALER_TAB_KEYS, value=USCALER_TAB_KEYS[5])
                upscaler_size_tab = gr.Slider(minimum=1., maximum=4., step=0.1, value=1.1, label="Upscale by")
                generate_button_up_tab = gr.Button(value="START UPSCALE", variant="primary")
            with gr.Column():
                result_up_tab = gr.Image(label="Result", type="pil", interactive=False, format="png")
                generate_button_up_tab.click(
                    fn=process_upscale,
                    inputs=[image_up_tab, upscaler_tab, upscaler_size_tab],
                    outputs=[result_up_tab],
                )

    with gr.Tab("Preprocessor", render=True):
        preprocessor_tab()

    gr.LoginButton()
    
if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True, share=True, debug=False, ssr_mode=False, mcp_server=False)
