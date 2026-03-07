import spaces
import os
from argparse import ArgumentParser
from stablepy import (
    Model_Diffusers,
    SCHEDULE_TYPE_OPTIONS,
    SCHEDULE_PREDICTION_TYPE_OPTIONS,
    check_scheduler_compatibility,
    TASK_AND_PREPROCESSORS,
    FACE_RESTORATION_MODELS,
    PROMPT_WEIGHT_OPTIONS_PRIORITY,
    scheduler_names,
)
from constants import (
    DIRECTORY_UPSCALERS,
    TASK_STABLEPY,
    TASK_MODEL_LIST,
    UPSCALER_DICT_GUI,
    UPSCALER_KEYS,
    PROMPT_W_OPTIONS,
    WARNING_MSG_VAE,
    SDXL_TASK,
    MODEL_TYPE_TASK,
    POST_PROCESSING_SAMPLER,
    DIFFUSERS_CONTROLNET_MODEL,
    IP_MODELS,
    MODE_IP_OPTIONS,
    CACHE_HF_ROOT,
)
from stablepy.diffusers_vanilla.style_prompt_config import STYLE_NAMES
import torch
import re
import time
from PIL import ImageFile
from utils import (
    get_model_list,
    extract_parameters,
    get_model_type,
    extract_exif_data,
    create_mask_now,
    download_diffuser_repo,
    get_used_storage_gb,
    delete_model,
    progress_step_bar,
    html_template_message,
    escape_html,
    clear_hf_cache,
)
from image_processor import preprocessor_tab
from datetime import datetime
import gradio as gr
import logging
import diffusers
import warnings
from stablepy import logger
from diffusers import FluxPipeline
# import urllib.parse
import subprocess

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
if IS_ZERO_GPU:
    subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
IS_GPU_MODE = True if IS_ZERO_GPU else (True if torch.cuda.is_available() else False)
img_path = "./images/"
allowed_path = os.path.abspath(img_path)
delete_cache_time = (9600, 9600) if IS_ZERO_GPU else (86400, 86400)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

## BEGIN MOD
logging.getLogger("diffusers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")
logger.setLevel(logging.DEBUG)

from env import (
    HF_TOKEN, HF_READ_TOKEN, # to use only for private repos
    CIVITAI_API_KEY, HF_LORA_PRIVATE_REPOS1, HF_LORA_PRIVATE_REPOS2,
    HF_LORA_ESSENTIAL_PRIVATE_REPO, HF_VAE_PRIVATE_REPO,
    HF_SDXL_EMBEDS_NEGATIVE_PRIVATE_REPO, HF_SDXL_EMBEDS_POSITIVE_PRIVATE_REPO,
    DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS,
    DIRECTORY_EMBEDS_SDXL, DIRECTORY_EMBEDS_POSITIVE_SDXL,
    LOAD_DIFFUSERS_FORMAT_MODEL, DOWNLOAD_MODEL_LIST, DOWNLOAD_LORA_LIST,
    DOWNLOAD_VAE_LIST, DOWNLOAD_EMBEDS)

from modutils import (to_list, list_uniq, list_sub, get_model_id_list, get_tupled_embed_list,
                      get_tupled_model_list, get_lora_model_list, download_private_repo, download_things, download_link_model)

# - **Download Models**
download_model = ", ".join(DOWNLOAD_MODEL_LIST)
# - **Download VAEs**
download_vae = ", ".join(DOWNLOAD_VAE_LIST)
# - **Download LoRAs**
download_lora = ", ".join(DOWNLOAD_LORA_LIST)

#download_private_repo(HF_LORA_ESSENTIAL_PRIVATE_REPO, DIRECTORY_LORAS, True)
download_private_repo(HF_VAE_PRIVATE_REPO, DIRECTORY_VAES, False)

load_diffusers_format_model = list_uniq(LOAD_DIFFUSERS_FORMAT_MODEL + get_model_id_list())
## END MOD

directories = [DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS, DIRECTORY_UPSCALERS]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Download stuffs
for url in [url.strip() for url in download_model.split(',')]:
    download_things(DIRECTORY_MODELS, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in download_vae.split(',')]:
    download_things(DIRECTORY_VAES, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in download_lora.split(',')]:
    download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)

# Download Embeddings
for url_embed in DOWNLOAD_EMBEDS:
    if not os.path.exists(f"./embedings/{url_embed.split('/')[-1]}"):
        download_things(DIRECTORY_EMBEDS, url_embed, HF_TOKEN, CIVITAI_API_KEY)

# Build list models
embed_list = get_model_list(DIRECTORY_EMBEDS)
single_file_model_list = get_model_list(DIRECTORY_MODELS)
model_list = list_uniq(get_model_id_list() + LOAD_DIFFUSERS_FORMAT_MODEL + single_file_model_list)

## BEGIN MOD
lora_model_list = get_lora_model_list()
vae_model_list = get_model_list(DIRECTORY_VAES)
vae_model_list.insert(0, "BakedVAE")
vae_model_list.insert(0, "None")

download_private_repo(HF_SDXL_EMBEDS_NEGATIVE_PRIVATE_REPO, DIRECTORY_EMBEDS_SDXL, False)
download_private_repo(HF_SDXL_EMBEDS_POSITIVE_PRIVATE_REPO, DIRECTORY_EMBEDS_POSITIVE_SDXL, False)
embed_sdxl_list = get_model_list(DIRECTORY_EMBEDS_SDXL) + get_model_list(DIRECTORY_EMBEDS_POSITIVE_SDXL)

def get_embed_list(pipeline_name):
    return get_tupled_embed_list(embed_sdxl_list if pipeline_name == "StableDiffusionXLPipeline" else embed_list)
## END MOD

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

components = None
if IS_ZERO_GPU:
    flux_repo = "camenduru/FLUX.1-dev-diffusers"
    flux_pipe = FluxPipeline.from_pretrained(
        flux_repo,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )#.to("cuda")
    components = flux_pipe.components
    delete_model(flux_repo)

parser = ArgumentParser(description='DiffuseCraft: Create images from text prompts.', add_help=True)
parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
parser.add_argument('--theme', type=str, default="NoCrypt/miku", help='Set the theme (default: NoCrypt/miku)')
parser.add_argument("--ssr", action="store_true", help="Enable SSR (Server-Side Rendering)")
parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level (default: INFO)")
args = parser.parse_args()

logger.setLevel(
    "INFO" if IS_ZERO_GPU else getattr(logging, args.log_level.upper())
)

def lora_chk(lora_):
    if isinstance(lora_, str) and lora_.strip() not in ["", "None"]:
        return lora_
    return None

## BEGIN MOD
class GuiSD:
    def __init__(self, stream=True):
        self.model = None
        self.status_loading = False
        self.sleep_loading = 4
        self.last_load = datetime.now()
        self.inventory = []


        # Avoid duplicate downloads
        self.active_downloads = set()
        self.download_lock = threading.Lock()
    def update_storage_models(self, storage_floor_gb=24, required_inventory_for_purge=3):
        while get_used_storage_gb() > storage_floor_gb:
            if len(self.inventory) < required_inventory_for_purge:
                break
            removal_candidate = self.inventory.pop(0)
            delete_model(removal_candidate)

        # Cleanup after 60 seconds of inactivity
        lowPrioCleanup = max((datetime.now() - self.last_load).total_seconds(), 0) > 120
        if lowPrioCleanup and not self.status_loading and get_used_storage_gb(CACHE_HF_ROOT) > (storage_floor_gb * 2):
            print("Cleaning up Hugging Face cache...")
            clear_hf_cache()
            self.inventory = [
                m for m in self.inventory if os.path.exists(m)
            ]

    def update_inventory(self, model_name):
        if model_name not in single_file_model_list:
            self.inventory = [
                m for m in self.inventory if m != model_name
            ] + [model_name]
        print(self.inventory)

    def load_new_model(self, model_name, vae_model, task, controlnet_model, progress=gr.Progress(track_tqdm=True)):
        lock_key = model_name

        while True:
            with self.download_lock:
                if lock_key not in self.active_downloads:
                    self.active_downloads.add(lock_key)
                    break

            yield f"Waiting for existing download to finish: {model_name}..."
            time.sleep(1)

        try:


            # download link model > model_name
            if model_name.startswith("http"): #
                yield f"Downloading model: {model_name}"
                #model_name = download_things(DIRECTORY_MODELS, model_name, HF_TOKEN, CIVITAI_API_KEY)
                model_name, model_type = download_link_model(model_name, DIRECTORY_MODELS) #
                if not model_name:
                    raise ValueError("Error retrieving model information from URL")
                is_link_model = True #
            else: is_link_model = False #

            if IS_ZERO_GPU:
                self.update_storage_models()

            vae_model = vae_model if vae_model != "None" else None
            model_type = get_model_type(model_name) if not is_link_model else model_type #
            dtype_model = torch.bfloat16 if model_type == "FLUX" else torch.float16

            if not os.path.exists(model_name):
                logger.debug(f"model_name={model_name}, vae_model={vae_model}, task={task}, controlnet_model={controlnet_model}")
                _ = download_diffuser_repo(
                    repo_name=model_name,
                    model_type=model_type,
                    revision="main",
                    token=True,
                )

            self.update_inventory(model_name)

        finally:
            with self.download_lock:
                self.active_downloads.discard(lock_key)
        for i in range(68):
            if not self.status_loading:
                self.status_loading = True
                if i > 0:
                    time.sleep(self.sleep_loading)
                    print("Previous model ops...")
                break
            time.sleep(0.5)
            print(f"Waiting queue {i}")
            yield "Waiting queue"

        self.status_loading = True

        yield f"Loading model: {model_name}"

        if vae_model == "BakedVAE":
            vae_model = model_name
        elif vae_model:
            vae_type = "SDXL" if "sdxl" in vae_model.lower() else "SD 1.5"
            if model_type != vae_type:
                gr.Warning(WARNING_MSG_VAE)

        print("Loading model...")

        try:
            start_time = time.time()

            if self.model is None:
                self.model = Model_Diffusers(
                    base_model_id=model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                    device="cpu" if IS_ZERO_GPU else None,
                    env_components=components,
                )
                self.model.advanced_params(image_preprocessor_cuda_active=IS_GPU_MODE)
            else:
                if self.model.base_model_id != model_name:
                    load_now_time = datetime.now()
                    elapsed_time = max((load_now_time - self.last_load).total_seconds(), 0)

                    if elapsed_time <= 9:
                        print("Waiting for the previous model's time ops...")
                        time.sleep(9 - elapsed_time)

                if IS_ZERO_GPU:
                    self.model.device = torch.device("cpu")
                self.model.load_pipe(
                    model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                )

            end_time = time.time()
            self.sleep_loading = max(min(int(end_time - start_time), 10), 4)
        except Exception as e:
            self.last_load = datetime.now()
            self.status_loading = False
            self.sleep_loading = 4
            raise e

        self.last_load = datetime.now()
        self.status_loading = False

        yield f"Model loaded: {model_name}"

    #@spaces.GPU
    @torch.inference_mode()
    def generate_pipeline(
        self,
        prompt,
        neg_prompt,
        num_images,
        steps,
        cfg,
        clip_skip,
        seed,
        lora1,
        lora_scale1,
        lora2,
        lora_scale2,
        lora3,
        lora_scale3,
        lora4,
        lora_scale4,
        lora5,
        lora_scale5,
        lora6,
        lora_scale6,
        lora7,
        lora_scale7,
        sampler,
        schedule_type,
        schedule_prediction_type,
        img_height,
        img_width,
        model_name,
        vae_model,
        task,
        image_control,
        preprocessor_name,
        preprocess_resolution,
        image_resolution,
        style_prompt,  # list []
        style_json_file,
        image_mask,
        strength,
        low_threshold,
        high_threshold,
        value_threshold,
        distance_threshold,
        recolor_gamma_correction,
        tile_blur_sigma,
        controlnet_output_scaling_in_unet,
        controlnet_start_threshold,
        controlnet_stop_threshold,
        textual_inversion,
        syntax_weights,
        upscaler_model_path,
        upscaler_increases_size,
        upscaler_tile_size,
        upscaler_tile_overlap,
        hires_steps,
        hires_denoising_strength,
        hires_sampler,
        hires_prompt,
        hires_negative_prompt,
        hires_before_adetailer,
        hires_after_adetailer,
        hires_schedule_type,
        hires_guidance_scale,
        controlnet_model,
        loop_generation,
        leave_progress_bar,
        disable_progress_bar,
        image_previews,
        display_images,
        save_generated_images,
        filename_pattern,
        image_storage_location,
        retain_compel_previous_load,
        retain_detailfix_model_previous_load,
        retain_hires_model_previous_load,
        t2i_adapter_preprocessor,
        t2i_adapter_conditioning_scale,
        t2i_adapter_conditioning_factor,
        enable_live_preview,
        freeu,
        generator_in_cpu,
        adetailer_inpaint_only,
        adetailer_verbose,
        adetailer_sampler,
        adetailer_active_a,
        prompt_ad_a,
        negative_prompt_ad_a,
        strength_ad_a,
        face_detector_ad_a,
        person_detector_ad_a,
        hand_detector_ad_a,
        mask_dilation_a,
        mask_blur_a,
        mask_padding_a,
        adetailer_active_b,
        prompt_ad_b,
        negative_prompt_ad_b,
        strength_ad_b,
        face_detector_ad_b,
        person_detector_ad_b,
        hand_detector_ad_b,
        mask_dilation_b,
        mask_blur_b,
        mask_padding_b,
        retain_task_cache_gui,
        guidance_rescale,
        image_ip1,
        mask_ip1,
        model_ip1,
        mode_ip1,
        scale_ip1,
        image_ip2,
        mask_ip2,
        model_ip2,
        mode_ip2,
        scale_ip2,
        pag_scale,
        face_restoration_model,
        face_restoration_visibility,
        face_restoration_weight,
    ):
        info_state = html_template_message("Navigating latent space...")
        yield info_state, gr.update(), gr.update()

        vae_model = vae_model if vae_model != "None" else None
        loras_list = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
        vae_msg = f"VAE: {vae_model}" if vae_model else ""
        msg_lora = ""

## BEGIN MOD
        loras_list = [s if s else "None" for s in loras_list]
        global lora_model_list
        lora_model_list = get_lora_model_list()
## END MOD
        
        logger.debug(f"Config model: {model_name}, {vae_model}, {loras_list}")

        task = TASK_STABLEPY[task]

        params_ip_img = []
        params_ip_msk = []
        params_ip_model = []
        params_ip_mode = []
        params_ip_scale = []

        all_adapters = [
            (image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1),
            (image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2),
        ]

        if not hasattr(self.model.pipe, "transformer"):
            for imgip, mskip, modelip, modeip, scaleip in all_adapters:
                if imgip:
                    params_ip_img.append(imgip)
                    if mskip:
                        params_ip_msk.append(mskip)
                    params_ip_model.append(modelip)
                    params_ip_mode.append(modeip)
                    params_ip_scale.append(scaleip)

        concurrency = 5
        self.model.stream_config(concurrency=concurrency, latent_resize_by=1, vae_decoding=False)

        if task != "txt2img" and not image_control:
            raise ValueError("Reference image is required. Please upload one in 'Image ControlNet/Inpaint/Img2img'.")

        if task in ["inpaint", "repaint"] and not image_mask:
            raise ValueError("Mask image not found. Upload one in 'Image Mask' to proceed.")

        if "https://" not in str(UPSCALER_DICT_GUI[upscaler_model_path]):
            upscaler_model = upscaler_model_path
        else:
            url_upscaler = UPSCALER_DICT_GUI[upscaler_model_path]

            if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"):
                download_things(DIRECTORY_UPSCALERS, url_upscaler, HF_TOKEN)

            upscaler_model = f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"

        logging.getLogger("ultralytics").setLevel(logging.INFO if adetailer_verbose else logging.ERROR)

        adetailer_params_A = {
            "face_detector_ad": face_detector_ad_a,
            "person_detector_ad": person_detector_ad_a,
            "hand_detector_ad": hand_detector_ad_a,
            "prompt": prompt_ad_a,
            "negative_prompt": negative_prompt_ad_a,
            "strength": strength_ad_a,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_a,
            "mask_blur": mask_blur_a,
            "mask_padding": mask_padding_a,
            "inpaint_only": adetailer_inpaint_only,
            "sampler": adetailer_sampler,
        }

        adetailer_params_B = {
            "face_detector_ad": face_detector_ad_b,
            "person_detector_ad": person_detector_ad_b,
            "hand_detector_ad": hand_detector_ad_b,
            "prompt": prompt_ad_b,
            "negative_prompt": negative_prompt_ad_b,
            "strength": strength_ad_b,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_b,
            "mask_blur": mask_blur_b,
            "mask_padding": mask_padding_b,
        }
        pipe_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "img_height": img_height,
            "img_width": img_width,
            "num_images": num_images,
            "num_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "pag_scale": float(pag_scale),
            "seed": seed,
            "image": image_control,
            "preprocessor_name": preprocessor_name,
            "preprocess_resolution": preprocess_resolution,
            "image_resolution": image_resolution,
            "style_prompt": style_prompt if style_prompt else "",
            "style_json_file": "",
            "image_mask": image_mask,  # only for Inpaint
            "strength": strength,  # only for Inpaint or ...
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "value_threshold": value_threshold,
            "distance_threshold": distance_threshold,
            "recolor_gamma_correction": float(recolor_gamma_correction),
            "tile_blur_sigma": int(tile_blur_sigma),
            "lora_A": lora_chk(lora1),
            "lora_scale_A": lora_scale1,
            "lora_B": lora_chk(lora2),
            "lora_scale_B": lora_scale2,
            "lora_C": lora_chk(lora3),
            "lora_scale_C": lora_scale3,
            "lora_D": lora_chk(lora4),
            "lora_scale_D": lora_scale4,
            "lora_E": lora_chk(lora5),
            "lora_scale_E": lora_scale5,
            "lora_F": lora_chk(lora6),
            "lora_scale_F": lora_scale6,
            "lora_G": lora_chk(lora7),
            "lora_scale_G": lora_scale7,
## BEGIN MOD
            "textual_inversion": get_embed_list(self.model.class_name) if textual_inversion else [],
## END MOD
            "syntax_weights": syntax_weights,  # "Classic"
            "sampler": sampler,
            "schedule_type": schedule_type,
            "schedule_prediction_type": schedule_prediction_type,
            "xformers_memory_efficient_attention": False,
            "gui_active": True,
            "loop_generation": loop_generation,
            "controlnet_conditioning_scale": float(controlnet_output_scaling_in_unet),
            "control_guidance_start": float(controlnet_start_threshold),
            "control_guidance_end": float(controlnet_stop_threshold),
            "generator_in_cpu": generator_in_cpu,
            "FreeU": freeu,
            "adetailer_A": adetailer_active_a,
            "adetailer_A_params": adetailer_params_A,
            "adetailer_B": adetailer_active_b,
            "adetailer_B_params": adetailer_params_B,
            "leave_progress_bar": leave_progress_bar,
            "disable_progress_bar": disable_progress_bar,
            "image_previews": image_previews,
            "display_images": False,
            "save_generated_images": save_generated_images,
            "filename_pattern": filename_pattern,
            "image_storage_location": image_storage_location,
            "retain_compel_previous_load": retain_compel_previous_load,
            "retain_detailfix_model_previous_load": retain_detailfix_model_previous_load,
            "retain_hires_model_previous_load": retain_hires_model_previous_load,
            "t2i_adapter_preprocessor": t2i_adapter_preprocessor,
            "t2i_adapter_conditioning_scale": float(t2i_adapter_conditioning_scale),
            "t2i_adapter_conditioning_factor": float(t2i_adapter_conditioning_factor),
            "upscaler_model_path": upscaler_model,
            "upscaler_increases_size": upscaler_increases_size,
            "upscaler_tile_size": upscaler_tile_size,
            "upscaler_tile_overlap": upscaler_tile_overlap,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_prompt": hires_prompt,
            "hires_negative_prompt": hires_negative_prompt,
            "hires_sampler": hires_sampler,
            "hires_before_adetailer": hires_before_adetailer,
            "hires_after_adetailer": hires_after_adetailer,
            "hires_schedule_type": hires_schedule_type,
            "hires_guidance_scale": hires_guidance_scale,
            "ip_adapter_image": params_ip_img,
            "ip_adapter_mask": params_ip_msk,
            "ip_adapter_model": params_ip_model,
            "ip_adapter_mode": params_ip_mode,
            "ip_adapter_scale": params_ip_scale,
            "face_restoration_model": face_restoration_model,
            "face_restoration_visibility": face_restoration_visibility,
            "face_restoration_weight": face_restoration_weight,
        }

        # kwargs for diffusers pipeline
        if guidance_rescale:
            pipe_params["guidance_rescale"] = guidance_rescale
        if IS_ZERO_GPU:
            self.model.device = torch.device("cuda:0")
            if hasattr(self.model.pipe, "transformer") and loras_list != ["None"] * self.model.num_loras:
                self.model.pipe.transformer.to(self.model.device)
                logger.debug("transformer to cuda")

        actual_progress = 0
        info_images = gr.update()
        for img, [seed, image_path, metadata] in self.model(**pipe_params):
            info_state = progress_step_bar(actual_progress, steps)
            actual_progress += concurrency
            if image_path:
                info_images = f"Seeds: {str(seed)}"
                if vae_msg:
                    info_images = info_images + "<br>" + vae_msg

                if "Cannot copy out of meta tensor; no data!" in self.model.last_lora_error:
                    msg_ram = "Unable to process the LoRAs due to high RAM usage; please try again later."
                    print(msg_ram)
                    msg_lora += f"<br>{msg_ram}"

                for status, lora in zip(self.model.lora_status, self.model.lora_memory):
                    if status:
                        msg_lora += f"<br>Loaded: {lora}"
                    elif status is not None:
                        msg_lora += f"<br>Error with: {lora}"

                if msg_lora:
                    info_images += msg_lora

                info_images = info_images + "<br>" + "GENERATION DATA:<br>" + escape_html(metadata[-1]) + "<br>-------<br>"

                download_links = "<br>".join(
                    [
                        f'<a href="{path.replace("/images/", f"/gradio_api/file={allowed_path}/")}" download="{os.path.basename(path)}">Download Image {i + 1}</a>'
                        for i, path in enumerate(image_path)
                    ]
                )
                if save_generated_images:
                    info_images += f"<br>{download_links}"
## BEGIN MOD
                if not display_images:
                    img = img if img else gr.update()

                info_state = "COMPLETE"

                if not isinstance(img, list): img = [img]
                img = save_images(img, metadata)
                img = [(i, None) for i in img]

            elif not enable_live_preview:
                img = [(gr.update(), None)]
## END MOD
            yield info_state, img, info_images
            #return info_state, img, info_images

def dynamic_gpu_duration(func, duration, *args):

    @spaces.GPU(duration=duration)
    def wrapped_func():
        yield from func(*args)

    return wrapped_func()


@spaces.GPU
def dummy_gpu():
    return None


def sd_gen_generate_pipeline(sd_gen, *args):
    gpu_duration_arg = int(args[-1]) if args[-1] else 59
    verbose_arg = int(args[-2])
    load_lora_cpu = args[-3]
    generation_args = args[:-3]
    lora_list = [
        None if item == "None" or item == "" else item # MOD
        for item in [args[7], args[9], args[11], args[13], args[15], args[17], args[19]]
    ]
    lora_status = [None] * sd_gen.model.num_loras

    msg_load_lora = "Updating LoRAs in GPU..."
    if load_lora_cpu:
        msg_load_lora = "Updating LoRAs in CPU..."

    if lora_list != sd_gen.model.lora_memory and lora_list != [None] * sd_gen.model.num_loras:
        yield msg_load_lora, gr.update(), gr.update()

    # Load lora in CPU
    if load_lora_cpu:
        lora_status = sd_gen.model.load_lora_on_the_fly(
            lora_A=lora_list[0], lora_scale_A=args[8],
            lora_B=lora_list[1], lora_scale_B=args[10],
            lora_C=lora_list[2], lora_scale_C=args[12],
            lora_D=lora_list[3], lora_scale_D=args[14],
            lora_E=lora_list[4], lora_scale_E=args[16],
            lora_F=lora_list[5], lora_scale_F=args[18],
            lora_G=lora_list[6], lora_scale_G=args[20],
        )
        print(lora_status)

    sampler_name = args[21]
    schedule_type_name = args[22]
    _, _, msg_sampler = check_scheduler_compatibility(
        sd_gen.model.class_name, sampler_name, schedule_type_name
    )
    if msg_sampler:
        gr.Warning(msg_sampler)

    if verbose_arg:
        for status, lora in zip(lora_status, lora_list):
            if status:
                gr.Info(f"LoRA loaded in CPU: {lora}")
            elif status is not None:
                gr.Warning(f"Failed to load LoRA: {lora}")

        if lora_status == [None] * sd_gen.model.num_loras and sd_gen.model.lora_memory != [None] * sd_gen.model.num_loras and load_lora_cpu:
            lora_cache_msg = ", ".join(
                str(x) for x in sd_gen.model.lora_memory if x is not None
            )
            gr.Info(f"LoRAs in cache: {lora_cache_msg}")

    msg_request = f"Requesting {gpu_duration_arg}s. of GPU time.\nModel: {sd_gen.model.base_model_id}"
    if verbose_arg:
        gr.Info(msg_request)
        print(msg_request)
    yield msg_request.replace("\n", "<br>"), gr.update(), gr.update()

    start_time = time.time()

    # yield from sd_gen.generate_pipeline(*generation_args)
    yield from dynamic_gpu_duration(
        sd_gen.generate_pipeline,
        gpu_duration_arg,
        *generation_args,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    msg_task_complete = (
        f"GPU task complete in: {int(round(execution_time, 0) + 1)} seconds"
    )

    if verbose_arg:
        gr.Info(msg_task_complete)
        print(msg_task_complete)

    yield msg_task_complete, gr.update(), gr.update()


@spaces.GPU(duration=15)
def process_upscale(image, upscaler_name, upscaler_size):
    if image is None:
        return None

    from stablepy.diffusers_vanilla.utils import save_pil_image_with_metadata
    from stablepy import load_upscaler_model

    image = image.convert("RGB")
    exif_image = extract_exif_data(image)

    name_upscaler = UPSCALER_DICT_GUI[upscaler_name]

    if "https://" in str(name_upscaler):

        if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{name_upscaler.split('/')[-1]}"):
            download_things(DIRECTORY_UPSCALERS, name_upscaler, HF_TOKEN)

        name_upscaler = f"./{DIRECTORY_UPSCALERS}/{name_upscaler.split('/')[-1]}"

    scaler_beta = load_upscaler_model(model=name_upscaler, tile=(0 if IS_ZERO_GPU else 192), tile_overlap=8, device=("cuda" if IS_GPU_MODE else "cpu"), half=IS_GPU_MODE)
    image_up = scaler_beta.upscale(image, upscaler_size, True)

    image_path = save_pil_image_with_metadata(image_up, f'{os.getcwd()}/up_images', exif_image)

    return image_path


# https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space/discussions/1#674969a022b99c122af5d407
# dynamic_gpu_duration.zerogpu = True
# sd_gen_generate_pipeline.zerogpu = True
#sd_gen = GuiSD()


from pathlib import Path
from PIL import Image
import PIL
import numpy as np
import random
import json
import shutil
import gc
import threading
from collections import defaultdict
from tagger.tagger import insert_model_recom_prompt
from modutils import (safe_float, escape_lora_basename, to_lora_key, to_lora_path, valid_model_name, set_textual_inversion_prompt,
    get_local_model_list, get_model_pipeline, get_private_lora_model_lists, get_valid_lora_name, get_state, set_state,
    get_valid_lora_path, get_valid_lora_wt, get_lora_info, CIVITAI_SORT, CIVITAI_PERIOD, CIVITAI_BASEMODEL,
    normalize_prompt_list, get_civitai_info, search_lora_on_civitai, translate_to_en, get_t2i_model_info, get_civitai_tag, save_image_history,
    get_all_lora_list, get_all_lora_tupled_list, update_lora_dict, download_lora, copy_lora, download_my_lora, set_prompt_loras,
    apply_lora_prompt, update_loras, search_civitai_lora, search_civitai_lora_json, update_civitai_selection, select_civitai_lora)


_PIPELINES = {}
_LOCKS = defaultdict(threading.Lock)


def get_gsd(repo_id, vae, task, controlnet_model):
    with _LOCKS[repo_id]:
        gsd = _PIPELINES.get(repo_id)
        if gsd is None:
            gsd = GuiSD()
            for _ in gsd.load_new_model(repo_id, vae, task, controlnet_model):
                pass
            _PIPELINES[repo_id] = gsd
        return gsd


#@spaces.GPU
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps,
          model_name=load_diffusers_format_model[0], lora1=None, lora1_wt=1.0, lora2=None, lora2_wt=1.0,
          lora3=None, lora3_wt=1.0, lora4=None, lora4_wt=1.0, lora5=None, lora5_wt=1.0, lora6=None, lora6_wt=1.0, lora7=None, lora7_wt=1.0,
          task=TASK_MODEL_LIST[0], prompt_syntax="Classic", sampler="Euler", vae=None, schedule_type=SCHEDULE_TYPE_OPTIONS[0], schedule_prediction_type=SCHEDULE_PREDICTION_TYPE_OPTIONS[0],
          clip_skip=True, pag_scale=0.0, free_u=False, guidance_rescale=0., image_control_dict=None, image_mask=None, strength=0.35, image_resolution=1024,
          controlnet_model=DIFFUSERS_CONTROLNET_MODEL[0], control_net_output_scaling=1.0, control_net_start_threshold=0., control_net_stop_threshold=1.,
          preprocessor_name="Canny", preprocess_resolution=512, low_threshold=100, high_threshold=200,
          value_threshold=0.1, distance_threshold=0.1, recolor_gamma_correction=1., tile_blur_sigma=9,
          image_ip1_dict=None, mask_ip1=None, model_ip1="plus_face", mode_ip1="original", scale_ip1=0.7,
          image_ip2_dict=None, mask_ip2=None, model_ip2="base", mode_ip2="style", scale_ip2=0.7,
          upscaler_model_path=None, upscaler_increases_size=1.0, upscaler_tile_size=0, upscaler_tile_overlap=8, hires_steps=30, hires_denoising_strength=0.55,
          hires_sampler="Use same sampler", hires_schedule_type="Use same schedule type", hires_guidance_scale=-1, hires_prompt="", hires_negative_prompt="",
          adetailer_inpaint_only=True, adetailer_verbose=False, adetailer_sampler="Use same sampler", adetailer_active_a=False,
          prompt_ad_a="", negative_prompt_ad_a="", strength_ad_a=0.35, face_detector_ad_a=True, person_detector_ad_a=True, hand_detector_ad_a=False,
          mask_dilation_a=4, mask_blur_a=4, mask_padding_a=32, adetailer_active_b=False, prompt_ad_b="", negative_prompt_ad_b="", strength_ad_b=0.35,
          face_detector_ad_b=True, person_detector_ad_b=True, hand_detector_ad_b=False, mask_dilation_b=4, mask_blur_b=4, mask_padding_b=32,
          active_textual_inversion=False, face_restoration_model=None, face_restoration_visibility=1., face_restoration_weight=.5,
          gpu_duration=59, translate=False, recom_prompt=True, progress=gr.Progress(track_tqdm=True)):
    MAX_SEED = np.iinfo(np.int32).max

    # FIX: Safely handle image_control_dict and image_ip1_dict/image_ip2_dict
    image_mask = None
    image_control = None
    mask_ip1 = None
    image_ip1 = None
    mask_ip2 = None
    image_ip2 = None
    
    # Handle image_control_dict (main control image)
    if isinstance(image_control_dict, dict):
        # Get background (main image)
        image_control = image_control_dict.get('background')
        # Safely get layers (mask) - check if it exists and is non-empty
        layers = image_control_dict.get('layers', [])
        if layers and len(layers) > 0:
            image_mask = layers[0] if not image_mask else image_mask
    
    # Override with explicit image_mask if provided
    if image_mask is None:
        image_mask = image_mask  # Keep as is
    
    # Handle image_ip1_dict (first IP adapter)
    if isinstance(image_ip1_dict, dict):
        image_ip1 = image_ip1_dict.get('background')
        layers_ip1 = image_ip1_dict.get('layers', [])
        if layers_ip1 and len(layers_ip1) > 0:
            mask_ip1 = layers_ip1[0] if not mask_ip1 else mask_ip1
    
    # Override with explicit mask_ip1 if provided
    if mask_ip1 is None:
        mask_ip1 = mask_ip1  # Keep as is
    
    # Handle image_ip2_dict (second IP adapter)
    if isinstance(image_ip2_dict, dict):
        image_ip2 = image_ip2_dict.get('background')
        layers_ip2 = image_ip2_dict.get('layers', [])
        if layers_ip2 and len(layers_ip2) > 0:
            mask_ip2 = layers_ip2[0] if not mask_ip2 else mask_ip2
    
    # Override with explicit mask_ip2 if provided
    if mask_ip2 is None:
        mask_ip2 = mask_ip2  # Keep as is
    
    style_prompt = None
    style_json = None
    hires_before_adetailer = False
    hires_after_adetailer = True
    loop_generation = 1
    leave_progress_bar = True
    disable_progress_bar = False
    image_previews = True
    display_images = False
    save_generated_images = False
    filename_pattern = "model,seed"
    image_storage_location = "./images"
    retain_compel_previous_load = False
    retain_detailfix_model_previous_load = False
    retain_hires_model_previous_load = False
    t2i_adapter_preprocessor = True
    adapter_conditioning_scale = 1
    adapter_conditioning_factor = 0.55
    enable_live_preview = True
    generator_in_cpu = False
    retain_task_cache = False
    load_lora_cpu = False
    verbose_info = False

    images: list[tuple[PIL.Image.Image, str | None]] = []
    progress(0, desc="Preparing...")

    if randomize_seed: 
        seed = random.randint(0, MAX_SEED)
    if seed > MAX_SEED: 
        seed = MAX_SEED
    generator = torch.Generator().manual_seed(seed).seed()

    if translate:
        prompt = translate_to_en(prompt)
        negative_prompt = translate_to_en(prompt)

    prompt, negative_prompt = insert_model_recom_prompt(prompt, negative_prompt, model_name, recom_prompt)
    progress(0.5, desc="Preparing...")
    lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt = \
        set_prompt_loras(prompt, prompt_syntax, model_name, lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt)
    lora1 = get_valid_lora_path(lora1)
    lora2 = get_valid_lora_path(lora2)
    lora3 = get_valid_lora_path(lora3)
    lora4 = get_valid_lora_path(lora4)
    lora5 = get_valid_lora_path(lora5)
    lora6 = get_valid_lora_path(lora6)
    lora7 = get_valid_lora_path(lora7)
    progress(1, desc="Preparation completed. Starting inference...")
    progress(0, desc="Loading model...")
    gsd = get_gsd(valid_model_name(model_name), vae, task, controlnet_model)
    progress(1, desc="Model loaded.")
    progress(0, desc="Starting Inference...")
    for info_state, stream_images, info_images in sd_gen_generate_pipeline(gsd, prompt, negative_prompt, 1, num_inference_steps,
        guidance_scale, clip_skip, generator, lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt,
        lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt, sampler, schedule_type, schedule_prediction_type,
        height, width, model_name, vae, task, image_control, preprocessor_name, preprocess_resolution, image_resolution,
        style_prompt, style_json, image_mask, strength, low_threshold, high_threshold, value_threshold, distance_threshold,
        recolor_gamma_correction, tile_blur_sigma, control_net_output_scaling, control_net_start_threshold, control_net_stop_threshold,
        active_textual_inversion, prompt_syntax, upscaler_model_path, upscaler_increases_size, upscaler_tile_size, upscaler_tile_overlap,
        hires_steps, hires_denoising_strength, hires_sampler, hires_prompt, hires_negative_prompt, hires_before_adetailer, hires_after_adetailer,
        hires_schedule_type, hires_guidance_scale, controlnet_model, loop_generation, leave_progress_bar, disable_progress_bar, image_previews,
        display_images, save_generated_images, filename_pattern, image_storage_location, retain_compel_previous_load, retain_detailfix_model_previous_load,
        retain_hires_model_previous_load, t2i_adapter_preprocessor, adapter_conditioning_scale, adapter_conditioning_factor, enable_live_preview,
        free_u, generator_in_cpu, adetailer_inpaint_only, adetailer_verbose, adetailer_sampler, adetailer_active_a, prompt_ad_a, negative_prompt_ad_a,
        strength_ad_a, face_detector_ad_a, person_detector_ad_a, hand_detector_ad_a, mask_dilation_a, mask_blur_a, mask_padding_a,
        adetailer_active_b, prompt_ad_b, negative_prompt_ad_b, strength_ad_b, face_detector_ad_b, person_detector_ad_b, hand_detector_ad_b,
        mask_dilation_b, mask_blur_b, mask_padding_b, retain_task_cache, guidance_rescale, image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1,
        image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2, pag_scale, face_restoration_model, face_restoration_visibility, face_restoration_weight,
        load_lora_cpu, verbose_info, gpu_duration
    ):
        images = stream_images if isinstance(stream_images, list) else images
    progress(1, desc="Inference completed.")
    output_image = images[0][0] if images else None

    gc.collect()

    return output_image


#@spaces.GPU
def _infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps,
           model_name=load_diffusers_format_model[0], lora1=None, lora1_wt=1.0, lora2=None, lora2_wt=1.0,
           lora3=None, lora3_wt=1.0, lora4=None, lora4_wt=1.0, lora5=None, lora5_wt=1.0, lora6=None, lora6_wt=1.0, lora7=None, lora7_wt=1.0,
           task=TASK_MODEL_LIST[0], prompt_syntax="Classic", sampler="Euler", vae=None, schedule_type=SCHEDULE_TYPE_OPTIONS[0], schedule_prediction_type=SCHEDULE_PREDICTION_TYPE_OPTIONS[0],
           clip_skip=True, pag_scale=0.0, free_u=False, guidance_rescale=0., image_control_dict=None, image_mask=None, strength=0.35, image_resolution=1024,
           controlnet_model=DIFFUSERS_CONTROLNET_MODEL[0], control_net_output_scaling=1.0, control_net_start_threshold=0., control_net_stop_threshold=1.,
           preprocessor_name="Canny", preprocess_resolution=512, low_threshold=100, high_threshold=200,
           value_threshold=0.1, distance_threshold=0.1, recolor_gamma_correction=1., tile_blur_sigma=9,
           image_ip1_dict=None, mask_ip1=None, model_ip1="plus_face", mode_ip1="original", scale_ip1=0.7,
           image_ip2_dict=None, mask_ip2=None, model_ip2="base", mode_ip2="style", scale_ip2=0.7,
           upscaler_model_path=None, upscaler_increases_size=1.0, upscaler_tile_size=0, upscaler_tile_overlap=8, hires_steps=30, hires_denoising_strength=0.55,
           hires_sampler="Use same sampler", hires_schedule_type="Use same schedule type", hires_guidance_scale=-1, hires_prompt="", hires_negative_prompt="",
           adetailer_inpaint_only=True, adetailer_verbose=False, adetailer_sampler="Use same sampler", adetailer_active_a=False,
           prompt_ad_a="", negative_prompt_ad_a="", strength_ad_a=0.35, face_detector_ad_a=True, person_detector_ad_a=True, hand_detector_ad_a=False,
           mask_dilation_a=4, mask_blur_a=4, mask_padding_a=32, adetailer_active_b=False, prompt_ad_b="", negative_prompt_ad_b="", strength_ad_b=0.35,
           face_detector_ad_b=True, person_detector_ad_b=True, hand_detector_ad_b=False, mask_dilation_b=4, mask_blur_b=4, mask_padding_b=32,
           active_textual_inversion=False, face_restoration_model=None, face_restoration_visibility=1., face_restoration_weight=.5,
           gpu_duration=59, translate=False, recom_prompt=True, progress=gr.Progress(track_tqdm=True)):
    return gr.update()


infer.zerogpu = True
_infer.zerogpu = True


def pass_result(result):
    return result


def get_samplers():
    return scheduler_names


def get_vaes():
    return vae_model_list


def update_task_options(model_name, task_name):
    new_choices = MODEL_TYPE_TASK[get_model_type(valid_model_name(model_name))]

    if task_name not in new_choices:
        task_name = "txt2img"

    return gr.update(value=task_name, choices=new_choices)


def change_preprocessor_choices(task):
    task = TASK_STABLEPY[task]
    if task in TASK_AND_PREPROCESSORS.keys():
        choices_task = TASK_AND_PREPROCESSORS[task]
    else:
        choices_task = TASK_AND_PREPROCESSORS["canny"]
    return gr.update(choices=choices_task, value=choices_task[0])


def get_ti_choices(model_name: str):
    return get_embed_list(get_model_pipeline(valid_model_name(model_name)))


def update_textual_inversion(active_textual_inversion: bool, model_name: str):
    return gr.update(choices=get_ti_choices(model_name) if active_textual_inversion else [])


cached_diffusers_model_tupled_list = get_tupled_model_list(load_diffusers_format_model)
def get_diffusers_model_list(state: dict = {}):
    show_diffusers_model_list_detail = get_state(state, "show_diffusers_model_list_detail")
    if show_diffusers_model_list_detail:
        return cached_diffusers_model_tupled_list
    else:
        return load_diffusers_format_model


def enable_diffusers_model_detail(is_enable: bool = False, model_name: str = "", state: dict = {}):
    show_diffusers_model_list_detail = is_enable
    new_value = model_name
    index = 0
    if model_name in set(load_diffusers_format_model):
        index = load_diffusers_format_model.index(model_name)
    if is_enable:
        new_value = cached_diffusers_model_tupled_list[index][1]
    else:
        new_value = load_diffusers_format_model[index]
    set_state(state, "show_diffusers_model_list_detail", show_diffusers_model_list_detail)
    return gr.update(value=is_enable), gr.update(value=new_value, choices=get_diffusers_model_list(state)), state


quality_prompt_list = [
    {
        "name": "None",
        "prompt": "",
        "negative_prompt": "lowres",
    },
    {
        "name": "Animagine Common",
        "prompt": "anime artwork, anime style, vibrant, studio anime, highly detailed, masterpiece, best quality, very aesthetic, absurdres",
        "negative_prompt": "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
    },
    {
        "name": "Pony Anime Common",
        "prompt": "source_anime, score_9, score_8_up, score_7_up, masterpiece, best quality, very aesthetic, absurdres",
        "negative_prompt": "source_pony, source_furry, source_cartoon, score_6, score_5, score_4, busty, ugly face, mutated hands, low res, blurry face, black and white, the simpsons, overwatch, apex legends",
    },
    {
        "name": "Pony Common",
        "prompt": "source_anime, score_9, score_8_up, score_7_up",
        "negative_prompt": "source_pony, source_furry, source_cartoon, score_6, score_5, score_4, busty, ugly face, mutated hands, low res, blurry face, black and white, the simpsons, overwatch, apex legends",
    },
    {
        "name": "Animagine Standard v3.0",
        "prompt": "masterpiece, best quality",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    },
    {
        "name": "Animagine Standard v3.1",
        "prompt": "masterpiece, best quality, very aesthetic, absurdres",
        "negative_prompt": "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
    },
    {
        "name": "Animagine Light v3.1",
        "prompt": "(masterpiece), best quality, very aesthetic, perfect face",
        "negative_prompt": "(low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn",
    },
    {
        "name": "Animagine Heavy v3.1",
        "prompt": "(masterpiece), (best quality), (ultra-detailed), very aesthetic, illustration, disheveled hair, perfect composition, moist skin, intricate details",
        "negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality, very displeasing",
    },
]


style_list = [
    {
        "name": "None",
        "prompt": "",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork, anime style, vibrant, studio anime, highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style, vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art, digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art, low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model, octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]


preset_styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
preset_quality = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in quality_prompt_list}


def process_style_prompt(prompt: str, neg_prompt: str, styles_key: str = "None", quality_key: str = "None"):
    def to_list(s):
        return [x.strip() for x in s.split(",") if not s == ""]
    
    def list_sub(a, b):
        return [e for e in a if e not in b]
    
    def list_uniq(l):
        return sorted(set(l), key=l.index)

    animagine_ps = to_list("anime artwork, anime style, vibrant, studio anime, highly detailed, masterpiece, best quality, very aesthetic, absurdres")
    animagine_nps = to_list("lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]")
    pony_ps = to_list("source_anime, score_9, score_8_up, score_7_up, masterpiece, best quality, very aesthetic, absurdres")
    pony_nps = to_list("source_pony, source_furry, source_cartoon, score_6, score_5, score_4, busty, ugly face, mutated hands, low res, blurry face, black and white, the simpsons, overwatch, apex legends")
    prompts = to_list(prompt)
    neg_prompts = to_list(neg_prompt)

    all_styles_ps = []
    all_styles_nps = []
    for d in style_list:
        all_styles_ps.extend(to_list(str(d.get("prompt", ""))))
        all_styles_nps.extend(to_list(str(d.get("negative_prompt", ""))))

    all_quality_ps = []
    all_quality_nps = []
    for d in quality_prompt_list:
        all_quality_ps.extend(to_list(str(d.get("prompt", ""))))
        all_quality_nps.extend(to_list(str(d.get("negative_prompt", ""))))

    quality_ps = to_list(preset_quality[quality_key][0])
    quality_nps = to_list(preset_quality[quality_key][1])
    styles_ps = to_list(preset_styles[styles_key][0])
    styles_nps = to_list(preset_styles[styles_key][1])

    prompts = list_sub(prompts, animagine_ps + pony_ps + all_styles_ps + all_quality_ps)
    neg_prompts = list_sub(neg_prompts, animagine_nps + pony_nps + all_styles_nps + all_quality_nps)

    last_empty_p = [""] if not prompts and type != "None" and type != "Auto" and styles_key != "None" and quality_key != "None" else []
    last_empty_np = [""] if not neg_prompts and type != "None" and type != "Auto" and styles_key != "None" and quality_key != "None" else []

    if type == "Animagine":
        prompts = prompts + animagine_ps
        neg_prompts = neg_prompts + animagine_nps
    elif type == "Pony":
        prompts = prompts + pony_ps
        neg_prompts = neg_prompts + pony_nps

    prompts = prompts + styles_ps + quality_ps
    neg_prompts = neg_prompts + styles_nps + quality_nps

    prompt = ", ".join(list_uniq(prompts) + last_empty_p)
    neg_prompt = ", ".join(list_uniq(neg_prompts) + last_empty_np)

    return gr.update(value=prompt), gr.update(value=neg_prompt)


def save_images(images: list[Image.Image], metadatas: list[str]):
    from PIL import PngImagePlugin
    import uuid
    try:
        output_images = []
        for image, metadata in zip(images, metadatas):
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", metadata)
            savefile = f"{str(uuid.uuid4())}.png"
            image.save(savefile, "PNG", pnginfo=info)
            output_images.append(str(Path(savefile).resolve()))
        return output_images
    except Exception as e:
        print(f"Failed to save image file: {e}")
        raise Exception(f"Failed to save image file:") from e
