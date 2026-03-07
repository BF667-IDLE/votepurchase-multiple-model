import spaces
import json
import gradio as gr
import os
import re
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib.parse
import pandas as pd
from typing import Any
from huggingface_hub import HfApi, HfFolder, hf_hub_download, snapshot_download
from translatepy import Translator
from unidecode import unidecode
import copy
from datetime import datetime, timezone, timedelta
FILENAME_TIMEZONE = timezone(timedelta(hours=9)) # JST
import torch
from safetensors.torch import load_file
import gc


from env import (HF_LORA_PRIVATE_REPOS1, HF_LORA_PRIVATE_REPOS2,
    HF_MODEL_USER_EX, HF_MODEL_USER_LIKES, DIFFUSERS_FORMAT_LORAS,
    DIRECTORY_LORAS, HF_READ_TOKEN, HF_TOKEN, CIVITAI_API_KEY)


MODEL_TYPE_DICT = {
    "diffusers:StableDiffusionPipeline": "SD 1.5",
    "diffusers:StableDiffusionXLPipeline": "SDXL",
    "diffusers:FluxPipeline": "FLUX",
}


def get_user_agent():
    return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'


def to_list(s):
    return [x.strip() for x in s.split(",") if not s == ""]


def list_uniq(l):
    return sorted(set(l), key=l.index)


def list_sub(a, b):
    return [e for e in a if e not in b]


def is_repo_name(s):
    return re.fullmatch(r'^[^/]+?/[^/]+?$', s)


DEFAULT_STATE = {
    "show_diffusers_model_list_detail": False,
}


def get_state(state: dict, key: str):
    if key in state.keys(): return state[key]
    elif key in DEFAULT_STATE.keys():
        print(f"State '{key}' not found. Use dedault value.")
        return DEFAULT_STATE[key]
    else:
        print(f"State '{key}' not found.")
        return None


def set_state(state: dict, key: str, value: Any):
    state[key] = value


translator = Translator()
def translate_to_en(input: str):
    try:
        output = str(translator.translate(input, 'English'))
    except Exception as e:
        output = input
        print(e)
    return output


def get_local_model_list(dir_path):
    model_list = []
    valid_extensions = ('.ckpt', '.pt', '.pth', '.safetensors', '.bin')
    for file in Path(dir_path).glob("*"):
        if file.suffix in valid_extensions:
            file_path = str(Path(f"{dir_path}/{file.name}"))
            model_list.append(file_path)
            #print('\033[34mFILE: ' + file_path + '\033[0m')
    return model_list


def get_token():
    try:
        token = HfFolder.get_token()
    except Exception:
        token = ""
    return token


def set_token(token):
    try:
        HfFolder.save_token(token)
    except Exception:
        print(f"Error: Failed to save token.")


set_token(HF_TOKEN)


def split_hf_url(url: str):
    try:
        s = list(re.findall(r'^(?:https?://huggingface.co/)(?:(datasets)/)?(.+?/.+?)/\w+?/.+?/(?:(.+)/)?(.+?.\w+)(?:\?download=true)?$', url)[0])
        if len(s) < 4: return "", "", "", ""
        repo_id = s[1]
        repo_type = "dataset" if s[0] == "datasets" else "model"
        subfolder = urllib.parse.unquote(s[2]) if s[2] else None
        filename = urllib.parse.unquote(s[3])
        return repo_id, filename, subfolder, repo_type
    except Exception as e:
        print(e)


def download_hf_file(directory, url, force_filename="", hf_token="", progress=gr.Progress(track_tqdm=True)):
    repo_id, filename, subfolder, repo_type = split_hf_url(url)
    kwargs = {}
    if subfolder is not None: kwargs["subfolder"] = subfolder
    if force_filename: kwargs["force_filename"] = force_filename
    try:
        print(f"Start downloading: {url} to {directory}")
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=directory, token=hf_token, **kwargs)
        return path
    except Exception as e:
        print(f"Download failed: {url} {e}")
        return None


USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'


def request_json_data(url):
    model_version_id = url.split('/')[-1]
    if "?modelVersionId=" in model_version_id:
        match = re.search(r'modelVersionId=(\d+)', url)
        model_version_id = match.group(1)

    endpoint_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"

    params = {}
    headers = {'User-Agent': USER_AGENT, 'content-type': 'application/json'}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        result = session.get(endpoint_url, params=params, headers=headers, stream=True, timeout=(3.0, 15))
        result.raise_for_status()
        json_data = result.json()
        return json_data if json_data else None
    except Exception as e:
        print(f"Error: {e}")
        return None


class ModelInformation:
    def __init__(self, json_data):
        self.model_version_id = json_data.get("id", "")
        self.model_id = json_data.get("modelId", "")
        self.download_url = json_data.get("downloadUrl", "")
        self.model_url = f"https://civitai.com/models/{self.model_id}?modelVersionId={self.model_version_id}"
        self.filename_url = next(
            (v.get("name", "") for v in reversed(json_data.get("files", [])) if str(self.model_version_id) in v.get("downloadUrl", "")), ""
        )
        self.filename_url = self.filename_url if self.filename_url else ""
        self.description = json_data.get("description", "")
        if self.description is None: self.description = ""
        self.model_name = json_data.get("model", {}).get("name", "")
        self.model_type = json_data.get("model", {}).get("type", "")
        self.nsfw = json_data.get("model", {}).get("nsfw", False)
        self.poi = json_data.get("model", {}).get("poi", False)
        self.images = [img.get("url", "") for img in json_data.get("images", [])]
        self.example_prompt = json_data.get("trainedWords", [""])[0] if json_data.get("trainedWords") else ""
        self.original_json = copy.deepcopy(json_data)


def retrieve_model_info(url):
    json_data = request_json_data(url)
    if not json_data:
        return None
    model_descriptor = ModelInformation(json_data)
    return model_descriptor


def download_things(directory, url, hf_token="", civitai_api_key="", romanize=False):
    hf_token = get_token()
    url = url.strip()
    downloaded_file_path = None

    if "drive.google.com" in url:
        original_dir = os.getcwd()
        os.chdir(directory)
        os.system(f"gdown --fuzzy {url}")
        os.chdir(original_dir)
    elif "huggingface.co" in url:
        url = url.replace("?download=true", "")
        # url = urllib.parse.quote(url, safe=':/')  # fix encoding
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")

        filename = unidecode(url.split('/')[-1]) if romanize else url.split('/')[-1]

        download_hf_file(directory, url, filename, hf_token)

        downloaded_file_path = os.path.join(directory, filename)

    elif "civitai.com" in url:

        if not civitai_api_key:
            print("\033[91mYou need an API key to download Civitai models.\033[0m")

        model_profile = retrieve_model_info(url)
        if model_profile.download_url and model_profile.filename_url:
            url = model_profile.download_url
            filename = unidecode(model_profile.filename_url) if romanize else model_profile.filename_url
        else:
            if "?" in url:
                url = url.split("?")[0]
            filename = ""

        url_dl = url + f"?token={civitai_api_key}"
        print(f"Filename: {filename}")

        param_filename = ""
        if filename:
            param_filename = f"-o '{filename}'"

        aria2_command = (
            f'aria2c --console-log-level=error --summary-interval=10 -c -x 16 '
            f'-k 1M -s 16 -d "{directory}" {param_filename} "{url_dl}"'
        )
        os.system(aria2_command)

        if param_filename and os.path.exists(os.path.join(directory, filename)):
            downloaded_file_path = os.path.join(directory, filename)

    else:
        os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} {url}")

    return downloaded_file_path


def get_download_file(temp_dir, url, civitai_key="", progress=gr.Progress(track_tqdm=True)):
    if not "http" in url and is_repo_name(url) and not Path(url).exists():
        print(f"Use HF Repo: {url}")
        new_file = url
    elif not "http" in url and Path(url).exists():
        print(f"Use local file: {url}")
        new_file = url
    elif Path(f"{temp_dir}/{url.split('/')[-1]}").exists():
        print(f"File to download alreday exists: {url}")
        new_file = f"{temp_dir}/{url.split('/')[-1]}"
    else:
        print(f"Start downloading: {url}")
        before = get_local_model_list(temp_dir)
        try:
            download_things(temp_dir, url.strip(), HF_TOKEN, civitai_key)
        except Exception:
            print(f"Download failed: {url}")
            return ""
        after = get_local_model_list(temp_dir)
        new_file = list_sub(after, before)[0] if list_sub(after, before) else ""
    if not new_file:
        print(f"Download failed: {url}")
        return ""
    print(f"Download completed: {url}")
    return new_file


def escape_lora_basename(basename: str):
    return basename.replace(".", "_").replace(" ", "_").replace(",", "")


def to_lora_key(path: str):
    return escape_lora_basename(Path(path).stem)


def to_lora_path(key: str):
    if Path(key).is_file(): return key
    path = Path(f"{DIRECTORY_LORAS}/{escape_lora_basename(key)}.safetensors")
    return str(path)


def safe_float(input):
    output = 1.0
    try:
        output = float(input)
    except Exception:
        output = 1.0
    return output


def valid_model_name(model_name: str):
    return model_name.split(" ")[0]


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


def save_gallery_images(images, model_name="", progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Updating gallery...")
    basename = f"{model_name.split('/')[-1]}_{datetime.now(FILENAME_TIMEZONE).strftime('%Y%m%d_%H%M%S')}_"
    if not images: return images, gr.update()
    output_images = []
    output_paths = []
    for i, image in enumerate(images):
        filename = f"{basename}{str(i + 1)}.png"
        oldpath = Path(image[0])
        newpath = oldpath
        try:
            if oldpath.exists():
                newpath = oldpath.resolve().rename(Path(filename).resolve())
        except Exception as e:
            print(e)
        finally: 
            output_paths.append(str(newpath))
            output_images.append((str(newpath), str(filename)))
    progress(1, desc="Gallery updated.")
    return gr.update(value=output_images), gr.update(value=output_paths, visible=True)


def save_gallery_history(images, files, history_gallery, history_files, progress=gr.Progress(track_tqdm=True)):
    if not images or not files: return gr.update(), gr.update()
    if not history_gallery: history_gallery = []
    if not history_files: history_files = []
    output_gallery = images + history_gallery
    output_files = files + history_files
    return gr.update(value=output_gallery), gr.update(value=output_files, visible=True)


def save_image_history(image, gallery, files, model_name: str, progress=gr.Progress(track_tqdm=True)):
    if not gallery: gallery = []
    if not files: files = []
    try:
        basename = f"{model_name.split('/')[-1]}_{datetime.now(FILENAME_TIMEZONE).strftime('%Y%m%d_%H%M%S')}"
        if image is None or not isinstance(image, (str, Image.Image, np.ndarray, tuple)): return gr.update(), gr.update()
        filename = f"{basename}.png"
        if isinstance(image, tuple): image = image[0]
        if isinstance(image, str): oldpath = image
        elif isinstance(image, Image.Image):
            oldpath = "temp.png"
            image.save(oldpath)
        elif isinstance(image, np.ndarray):
            oldpath = "temp.png"
            Image.fromarray(image).convert('RGBA').save(oldpath)
        oldpath = Path(oldpath)
        newpath = oldpath
        if oldpath.exists():
            shutil.copy(oldpath.resolve(), Path(filename).resolve())
            newpath = Path(filename).resolve()
        files.insert(0, str(newpath))
        gallery.insert(0, (str(newpath), str(filename)))
    except Exception as e:
        print(e)
    finally: 
        return gr.update(value=gallery), gr.update(value=files, visible=True)


def download_private_repo(repo_id, dir_path, is_replace):
    if not HF_READ_TOKEN: return
    try:
        snapshot_download(repo_id=repo_id, local_dir=dir_path, allow_patterns=['*.ckpt', '*.pt', '*.pth', '*.safetensors', '*.bin'], token=HF_READ_TOKEN)
    except Exception as e:
        print(f"Error: Failed to download {repo_id}.")
        print(e)
        return
    if is_replace:
        for file in Path(dir_path).glob("*"):
            if file.exists() and "." in file.stem or " " in file.stem and file.suffix in ['.ckpt', '.pt', '.pth', '.safetensors', '.bin']:
                newpath = Path(f'{file.parent.name}/{escape_lora_basename(file.stem)}{file.suffix}')
                file.resolve().rename(newpath.resolve())


private_model_path_repo_dict = {} # {"local filepath": "huggingface repo_id", ...}


def get_private_model_list(repo_id, dir_path):
    global private_model_path_repo_dict
    api = HfApi()
    if not HF_READ_TOKEN: return []
    try:
        files = api.list_repo_files(repo_id, token=HF_READ_TOKEN)
    except Exception as e:
        print(f"Error: Failed to list {repo_id}.")
        print(e)
        return []
    model_list = []
    for file in files:
        path = Path(f"{dir_path}/{file}")
        if path.suffix in ['.ckpt', '.pt', '.pth', '.safetensors', '.bin']:
            model_list.append(str(path))
    for model in model_list:
        private_model_path_repo_dict[model] = repo_id
    return model_list


def download_private_file(repo_id, path, is_replace):
    file = Path(path)
    newpath = Path(f'{file.parent.name}/{escape_lora_basename(file.stem)}{file.suffix}') if is_replace else file
    if not HF_READ_TOKEN or newpath.exists(): return
    filename = file.name
    dirname = file.parent.name
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dirname, token=HF_READ_TOKEN)
    except Exception as e:
        print(f"Error: Failed to download {filename}.")
        print(e)
        return
    if is_replace:
        file.resolve().rename(newpath.resolve())


def download_private_file_from_somewhere(path, is_replace):
    if not path in private_model_path_repo_dict.keys(): return
    repo_id = private_model_path_repo_dict.get(path, None)
    download_private_file(repo_id, path, is_replace)


model_id_list = []
def get_model_id_list():
    global model_id_list
    if len(model_id_list) != 0: return model_id_list
    api = HfApi()
    model_ids = []
    try:
        models_likes = []
        for author in HF_MODEL_USER_LIKES:
            models_likes.extend(api.list_models(author=author, task="text-to-image", cardData=True, sort="likes"))
        models_ex = []
        for author in HF_MODEL_USER_EX:
            models_ex = api.list_models(author=author, task="text-to-image", cardData=True, sort="last_modified")
    except Exception as e:
        print(f"Error: Failed to list {author}'s models.")
        print(e)
        return model_ids
    for model in models_likes:
        model_ids.append(model.id) if not model.private else ""
    anime_models = []
    real_models = []
    anime_models_flux = []
    real_models_flux = []
    for model in models_ex:
        if not model.private and not model.gated:
            if "diffusers:FluxPipeline" in model.tags: anime_models_flux.append(model.id) if "anime" in model.tags else real_models_flux.append(model.id)
            else: anime_models.append(model.id) if "anime" in model.tags else real_models.append(model.id)
    model_ids.extend(anime_models)
    model_ids.extend(real_models)
    model_ids.extend(anime_models_flux)
    model_ids.extend(real_models_flux)
    model_id_list = model_ids.copy()
    return model_ids


model_id_list = get_model_id_list()


def get_t2i_model_info(repo_id: str):
    api = HfApi(token=HF_TOKEN)
    try:
        if not is_repo_name(repo_id): return ""
        model = api.model_info(repo_id=repo_id, timeout=5.0)
    except Exception as e:
        print(f"Error: Failed to get {repo_id}'s info.")
        print(e)
        return ""
    if model.private or model.gated: return ""
    tags = model.tags
    info = []
    url = f"https://huggingface.co/{repo_id}/"
    if not 'diffusers' in tags: return ""
    for k, v in MODEL_TYPE_DICT.items():
        if k in tags: info.append(v)
    if model.card_data and model.card_data.tags:
        info.extend(list_sub(model.card_data.tags, ['text-to-image', 'stable-diffusion', 'stable-diffusion-api', 'safetensors', 'stable-diffusion-xl']))
    info.append(f"DLs: {model.downloads}")
    info.append(f"likes: {model.likes}")
    info.append(model.last_modified.strftime("lastmod: %Y-%m-%d"))
    md = f"Model Info: {', '.join(info)}, [Model Repo]({url})"
    return gr.update(value=md)


MAX_MODEL_INFO = 100


def get_tupled_model_list(model_list):
    if not model_list: return []
    #return [(x, x) for x in model_list] # for skipping this function
    tupled_list = []
    api = HfApi()
    for i, repo_id in enumerate(model_list):
        if i > MAX_MODEL_INFO:
            tupled_list.append((repo_id, repo_id))
            continue
        try:
            if not api.repo_exists(repo_id): continue
            model = api.model_info(repo_id=repo_id, timeout=0.5)
        except Exception as e:
            print(f"{repo_id}: {e}")
            tupled_list.append((repo_id, repo_id))
            continue
        if model.tags is None: continue
        tags = model.tags
        info = []
        if not 'diffusers' in tags: continue
        for k, v in MODEL_TYPE_DICT.items():
            if k in tags: info.append(v)
        if model.card_data and model.card_data.tags:
            info.extend(list_sub(model.card_data.tags, ['text-to-image', 'stable-diffusion', 'stable-diffusion-api', 'safetensors', 'stable-diffusion-xl']))
        if "pony" in info:
            info.remove("pony")
            name = f"{repo_id} (Pony🐴, {', '.join(info)})"
        else:
            name = f"{repo_id} ({', '.join(info)})"
        tupled_list.append((name, repo_id))
    return tupled_list


private_lora_dict = {}
try:
    with open('lora_dict.json', encoding='utf-8') as f:
        d = json.load(f)
        for k, v in d.items():
            private_lora_dict[escape_lora_basename(k)] = v
except Exception as e:
    print(e)
loras_dict = {"None": ["", "", "", "", ""], "": ["", "", "", "", ""]} | private_lora_dict.copy()
civitai_not_exists_list = []
loras_url_to_path_dict = {} # {"URL to download": "local filepath", ...}
civitai_last_results = {}  # {"URL to download": {search results}, ...}
civitai_last_choices = [("", "")]
civitai_last_gallery = []
all_lora_list = []


private_lora_model_list = []
def get_private_lora_model_lists():
    global private_lora_model_list
    if len(private_lora_model_list) != 0: return private_lora_model_list
    models1 = []
    models2 = []
    for repo in HF_LORA_PRIVATE_REPOS1:
        models1.extend(get_private_model_list(repo, DIRECTORY_LORAS))
    for repo in HF_LORA_PRIVATE_REPOS2:
        models2.extend(get_private_model_list(repo, DIRECTORY_LORAS))
    models = list_uniq(models1 + sorted(models2))
    private_lora_model_list = models.copy()
    return models


private_lora_model_list = get_private_lora_model_lists()


def get_civitai_info(path):
    global civitai_not_exists_list
    default = ["", "", "", "", ""]
    if path in set(civitai_not_exists_list): return default
    if not Path(path).exists(): return None
    user_agent = get_user_agent()
    headers = {'User-Agent': user_agent, 'content-type': 'application/json'}
    base_url = 'https://civitai.com/api/v1/model-versions/by-hash/'
    params = {}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    import hashlib
    with open(path, 'rb') as file:
        file_data = file.read()
    hash_sha256 = hashlib.sha256(file_data).hexdigest()
    url = base_url + hash_sha256
    try:
        r = session.get(url, params=params, headers=headers, stream=True, timeout=(3.0, 15))
    except Exception as e:
        print(e)
        return default
    if not r.ok: return None
    json = r.json()
    if not 'baseModel' in json:
        civitai_not_exists_list.append(path)
        return default
    items = []
    items.append(" / ".join(json['trainedWords']))
    items.append(json['baseModel'])
    items.append(json['model']['name'])
    items.append(f"https://civitai.com/models/{json['modelId']}")
    items.append(json['images'][0]['url'])
    return items


def get_lora_model_list():
    loras = list_uniq(get_private_lora_model_lists() + DIFFUSERS_FORMAT_LORAS + get_local_model_list(DIRECTORY_LORAS))
    loras.insert(0, "None")
    loras.insert(0, "")
    return loras


def get_all_lora_list():
    global all_lora_list
    loras = get_lora_model_list()
    all_lora_list = loras.copy()
    return loras


def get_all_lora_tupled_list():
    global loras_dict
    models = get_all_lora_list()
    if not models: return []
    tupled_list = []
    for model in models:
        #if not model: continue # to avoid GUI-related bug
        basename = Path(model).stem
        key = to_lora_key(model)
        items = None
        if key in loras_dict.keys():
            items = loras_dict.get(key, None)
        else:
            items = get_civitai_info(model)
            if items != None:
                loras_dict[key] = items
        name = basename
        value = model
        if items and items[2] != "":
            if items[1] == "Pony":
                name = f"{basename} (for {items[1]}🐴, {items[2]})"
            else:
                name = f"{basename} (for {items[1]}, {items[2]})"
        tupled_list.append((name, value))
    return tupled_list


def update_lora_dict(path):
    global loras_dict
    key = escape_lora_basename(Path(path).stem)
    if key in loras_dict.keys(): return
    items = get_civitai_info(path)
    if items == None: return
    loras_dict[key] = items


def download_lora(dl_urls: str):
    global loras_url_to_path_dict
    dl_path = ""
    before = get_local_model_list(DIRECTORY_LORAS)
    urls = []
    for url in [url.strip() for url in dl_urls.split(',')]:
        local_path = f"{DIRECTORY_LORAS}/{url.split('/')[-1]}"
        if not Path(local_path).exists():
            download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)
            urls.append(url)
    after = get_local_model_list(DIRECTORY_LORAS)
    new_files = list_sub(after, before)
    i = 0
    for file in new_files:
        path = Path(file)
        if path.exists():
            new_path = Path(f'{path.parent.name}/{escape_lora_basename(path.stem)}{path.suffix}')
            path.resolve().rename(new_path.resolve())
            loras_url_to_path_dict[urls[i]] = str(new_path)
            update_lora_dict(str(new_path))
            dl_path = str(new_path)
        i += 1
    return dl_path


def copy_lora(path: str, new_path: str):
    if path == new_path: return new_path
    cpath = Path(path)
    npath = Path(new_path)
    if cpath.exists():
        try:
            shutil.copy(str(cpath.resolve()), str(npath.resolve()))
        except Exception as e:
            print(e)
            return None
        update_lora_dict(str(npath))
        return new_path
    else:
        return None


def download_my_lora(dl_urls: str, lora1: str, lora2: str, lora3: str, lora4: str, lora5: str, lora6: str, lora7: str):
    path = download_lora(dl_urls)
    if path:
        if not lora1 or lora1 == "None":
            lora1 = path
        elif not lora2 or lora2 == "None":
            lora2 = path
        elif not lora3 or lora3 == "None":
            lora3 = path
        elif not lora4 or lora4 == "None":
            lora4 = path
        elif not lora5 or lora5 == "None":
            lora5 = path
        #elif not lora6 or lora6 == "None":
        #    lora6 = path
        #elif not lora7 or lora7 == "None":
        #    lora7 = path
    choices = get_all_lora_tupled_list()
    return gr.update(value=lora1, choices=choices), gr.update(value=lora2, choices=choices), gr.update(value=lora3, choices=choices),\
        gr.update(value=lora4, choices=choices), gr.update(value=lora5, choices=choices), gr.update(value=lora6, choices=choices), gr.update(value=lora7, choices=choices)


def get_valid_lora_name(query: str, model_name: str):
    path = "None"
    if not query or query == "None": return "None"
    if to_lora_key(query) in loras_dict.keys(): return query
    if query in loras_url_to_path_dict.keys():
        path = loras_url_to_path_dict[query]
    else:
        path = to_lora_path(query.strip().split('/')[-1])
    if Path(path).exists():
        return path
    elif "http" in query:
        dl_file = download_lora(query)
        if dl_file and Path(dl_file).exists(): return dl_file
    else:
        dl_file = find_similar_lora(query, model_name)
        if dl_file and Path(dl_file).exists(): return dl_file
    return "None"


def get_valid_lora_path(query: str):
    path = None
    if not query or query == "None": return None
    if to_lora_key(query) in loras_dict.keys(): return query
    if Path(path).exists():
        return path
    else:
        return None


def get_valid_lora_wt(prompt: str, lora_path: str, lora_wt: float):
    wt = lora_wt
    result = re.findall(f'<lora:{to_lora_key(lora_path)}:(.+?)>', prompt)
    if not result: return wt
    wt = safe_float(result[0][0])
    return wt


def set_prompt_loras(prompt, prompt_syntax, model_name, lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt):
    if not "Classic" in str(prompt_syntax):  return lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt
    lora1 = get_valid_lora_name(lora1, model_name)
    lora2 = get_valid_lora_name(lora2, model_name)
    lora3 = get_valid_lora_name(lora3, model_name)
    lora4 = get_valid_lora_name(lora4, model_name)
    lora5 = get_valid_lora_name(lora5, model_name)
    #lora6 = get_valid_lora_name(lora6, model_name)
    #lora7 = get_valid_lora_name(lora7, model_name)
    if not "<lora" in prompt: return lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt
    lora1_wt = get_valid_lora_wt(prompt, lora1, lora1_wt)
    lora2_wt = get_valid_lora_wt(prompt, lora2, lora2_wt)
    lora3_wt = get_valid_lora_wt(prompt, lora3, lora3_wt)
    lora4_wt = get_valid_lora_wt(prompt, lora4, lora4_wt)
    lora5_wt = get_valid_lora_wt(prompt, lora5, lora5_wt)
    #lora6_wt = get_valid_lora_wt(prompt, lora6, lora5_wt)
    #lora7_wt = get_valid_lora_wt(prompt, lora7, lora5_wt)
    on1, label1, tag1, md1 = get_lora_info(lora1)
    on2, label2, tag2, md2 = get_lora_info(lora2)
    on3, label3, tag3, md3 = get_lora_info(lora3)
    on4, label4, tag4, md4 = get_lora_info(lora4)
    on5, label5, tag5, md5 = get_lora_info(lora5)
    #on6, label6, tag6, md6 = get_lora_info(lora6)
    #on7, label7, tag7, md7 = get_lora_info(lora7)
    lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
    prompts = prompt.split(",") if prompt else []
    for p in prompts:
        p = str(p).strip()
        if "<lora" in p:
            result = re.findall(r'<lora:(.+?):(.+?)>', p)
            if not result: continue
            key = result[0][0]
            wt = result[0][1]
            path = to_lora_path(key)
            if not key in loras_dict.keys() or not Path(path).exists():
                path = get_valid_lora_name(path)
                if not path or path == "None": continue
            if path in lora_paths or key in lora_paths:
                continue
            elif not on1:
                lora1 = path
                lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
                lora1_wt = safe_float(wt)
                on1 = True
            elif not on2:
                lora2 = path
                lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
                lora2_wt = safe_float(wt)
                on2 = True
            elif not on3:
                lora3 = path
                lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
                lora3_wt = safe_float(wt)
                on3 = True
            elif not on4:
                lora4 = path
                lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
                lora4_wt = safe_float(wt)
                on4 = True
            elif not on5:
                lora5 = path
                lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
                lora5_wt = safe_float(wt)
                on5 = True
            #elif not on6:
            #    lora6 = path
            #    lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
            #    lora6_wt = safe_float(wt)
            #    on6 = True
            #elif not on7:
            #    lora7 = path
            #    lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
            #    lora7_wt = safe_float(wt)
            #    on7 = True
    return lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt


def get_lora_info(lora_path: str):
    is_valid = False
    tag = ""
    label = ""
    md = "None"
    if not lora_path or lora_path == "None":
        print("LoRA file not found.")
        return is_valid, label, tag, md
    path = Path(lora_path)
    new_path = Path(f'{path.parent.name}/{escape_lora_basename(path.stem)}{path.suffix}')
    if not to_lora_key(str(new_path)) in loras_dict.keys() and str(path) not in set(get_all_lora_list()):
        print("LoRA file is not registered.")
        return tag, label, tag, md
    if not new_path.exists():
        download_private_file_from_somewhere(str(path), True)
    basename = new_path.stem
    label = f'Name: {basename}'
    items = loras_dict.get(basename, None)
    if items == None:
        items = get_civitai_info(str(new_path))
        if items != None:
            loras_dict[basename] = items
    if items and items[2] != "":
        tag = items[0]
        label = f'Name: {basename}'
        if items[1] == "Pony":
            label = f'Name: {basename} (for Pony🐴)'
        if items[4]:
            md = f'<img src="{items[4]}" alt="thumbnail" width="150" height="240"><br>[LoRA Model URL]({items[3]})'
        elif items[3]:
            md = f'[LoRA Model URL]({items[3]})'
    is_valid = True
    return is_valid, label, tag, md


def normalize_prompt_list(tags: list[str]):
    prompts = []
    for tag in tags:
        tag = str(tag).strip()
        if tag:
            prompts.append(tag)
    return prompts


def apply_lora_prompt(prompt: str = "", lora_info: str = ""):
    if lora_info == "None": return gr.update(value=prompt)
    tags = prompt.split(",") if prompt else []
    prompts = normalize_prompt_list(tags)

    lora_tag = lora_info.replace("/",",")
    lora_tags = lora_tag.split(",") if str(lora_info) != "None" else []
    lora_prompts = normalize_prompt_list(lora_tags)
 
    empty = [""]
    prompt = ", ".join(list_uniq(prompts + lora_prompts) + empty)
    return gr.update(value=prompt)


def update_loras(prompt, prompt_syntax, lora1, lora1_wt, lora2, lora2_wt, lora3, lora3_wt, lora4, lora4_wt, lora5, lora5_wt, lora6, lora6_wt, lora7, lora7_wt):
    on1, label1, tag1, md1 = get_lora_info(lora1)
    on2, label2, tag2, md2 = get_lora_info(lora2)
    on3, label3, tag3, md3 = get_lora_info(lora3)
    on4, label4, tag4, md4 = get_lora_info(lora4)
    on5, label5, tag5, md5 = get_lora_info(lora5)
    on6, label6, tag6, md6 = get_lora_info(lora6)
    on7, label7, tag7, md7 = get_lora_info(lora7)
    lora_paths = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]

    output_prompt = prompt
    if "Classic" in str(prompt_syntax):
        prompts = prompt.split(",") if prompt else []
        output_prompts = []
        for p in prompts:
            p = str(p).strip()
            if "<lora" in p:
                result = re.findall(r'<lora:(.+?):(.+?)>', p)
                if not result: continue
                key = result[0][0]
                wt = result[0][1]
                path = to_lora_path(key)
                if not key in loras_dict.keys() or not path: continue
                if path in lora_paths:
                    output_prompts.append(f"<lora:{to_lora_key(path)}:{safe_float(wt):.2f}>")
            elif p:
                output_prompts.append(p)
        lora_prompts = []
        if on1: lora_prompts.append(f"<lora:{to_lora_key(lora1)}:{lora1_wt:.2f}>")
        if on2: lora_prompts.append(f"<lora:{to_lora_key(lora2)}:{lora2_wt:.2f}>")
        if on3: lora_prompts.append(f"<lora:{to_lora_key(lora3)}:{lora3_wt:.2f}>")
        if on4: lora_prompts.append(f"<lora:{to_lora_key(lora4)}:{lora4_wt:.2f}>")
        if on5: lora_prompts.append(f"<lora:{to_lora_key(lora5)}:{lora5_wt:.2f}>")
        #if on6: lora_prompts.append(f"<lora:{to_lora_key(lora6)}:{lora6_wt:.2f}>")
        #if on7: lora_prompts.append(f"<lora:{to_lora_key(lora7)}:{lora7_wt:.2f}>")
        output_prompt = ", ".join(list_uniq(output_prompts + lora_prompts + [""]))
    choices = get_all_lora_tupled_list()

    return gr.update(value=output_prompt), gr.update(value=lora1, choices=choices), gr.update(value=lora1_wt),\
     gr.update(value=tag1, label=label1, visible=on1), gr.update(visible=on1), gr.update(value=md1, visible=on1),\
     gr.update(value=lora2, choices=choices), gr.update(value=lora2_wt),\
     gr.update(value=tag2, label=label2, visible=on2), gr.update(visible=on2), gr.update(value=md2, visible=on2),\
     gr.update(value=lora3, choices=choices), gr.update(value=lora3_wt),\
     gr.update(value=tag3, label=label3, visible=on3), gr.update(visible=on3), gr.update(value=md3, visible=on3),\
     gr.update(value=lora4, choices=choices), gr.update(value=lora4_wt),\
     gr.update(value=tag4, label=label4, visible=on4), gr.update(visible=on4), gr.update(value=md4, visible=on4),\
     gr.update(value=lora5, choices=choices), gr.update(value=lora5_wt),\
     gr.update(value=tag5, label=label5, visible=on5), gr.update(visible=on5), gr.update(value=md5, visible=on5),\
     gr.update(value=lora6, choices=choices), gr.update(value=lora6_wt),\
     gr.update(value=tag6, label=label6, visible=on6), gr.update(visible=on6), gr.update(value=md6, visible=on6),\
     gr.update(value=lora7, choices=choices), gr.update(value=lora7_wt),\
     gr.update(value=tag7, label=label7, visible=on7), gr.update(visible=on7), gr.update(value=md7, visible=on7)


def get_my_lora(link_url, romanize):
    l_name = ""
    l_path = ""
    before = get_local_model_list(DIRECTORY_LORAS)
    for url in [url.strip() for url in link_url.split(',')]:
        if not Path(f"{DIRECTORY_LORAS}/{url.split('/')[-1]}").exists():
            l_name = download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY, romanize)
    after = get_local_model_list(DIRECTORY_LORAS)
    new_files = list_sub(after, before)
    for file in new_files:
        path = Path(file)
        if path.exists():
            new_path = Path(f'{path.parent.name}/{escape_lora_basename(path.stem)}{path.suffix}')
            path.resolve().rename(new_path.resolve())
            update_lora_dict(str(new_path))
            l_path = str(new_path)
    new_lora_tupled_list = get_all_lora_tupled_list()
    msg_lora = "Downloaded"
    if l_name:
        msg_lora += f": <b>{l_name}</b>"
        print(msg_lora)

    return gr.update(
        choices=new_lora_tupled_list, value=l_path
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        value=msg_lora
    )


def upload_file_lora(files, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Uploading...")
    file_paths = [file.name for file in files]
    progress(1, desc="Uploaded.")
    return gr.update(value=file_paths, visible=True), gr.update()


def move_file_lora(filepaths):
    for file in filepaths:
        path = Path(shutil.move(Path(file).resolve(), Path(f"./{DIRECTORY_LORAS}").resolve()))
        newpath = Path(f'{path.parent.name}/{escape_lora_basename(path.stem)}{path.suffix}')
        path.resolve().rename(newpath.resolve())
        update_lora_dict(str(newpath))

    new_lora_model_list = get_lora_model_list()
    new_lora_tupled_list = get_all_lora_tupled_list()
    
    return gr.update(
        choices=new_lora_tupled_list, value=new_lora_model_list[-1]
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    ), gr.update(
        choices=new_lora_tupled_list
    )


CIVITAI_SORT = ["Highest Rated", "Most Downloaded", "Most Liked", "Most Discussed", "Most Collected", "Most Buzz", "Newest"]
CIVITAI_PERIOD = ["AllTime", "Year", "Month", "Week", "Day"]
CIVITAI_BASEMODEL = ["Pony", "Illustrious", "SDXL 1.0", "SD 1.5", "Flux.1 D", "Flux.1 S"] # , "SD 3.5"
CIVITAI_TYPE = ["Checkpoint", "TextualInversion", "Hypernetwork", "AestheticGradient", "LORA", "LoCon", "DoRA",
                "Controlnet", "Upscaler", "MotionModule", "VAE", "Poses", "Wildcards", "Workflows", "Other"]
CIVITAI_FILETYPE = ["Model", "VAE", "Config", "Training Data"]


def get_civitai_info(path):
    global civitai_not_exists_list, loras_url_to_path_dict
    default = ["", "", "", "", ""]
    if path in set(civitai_not_exists_list): return default
    if not Path(path).exists(): return None
    user_agent = get_user_agent()
    headers = {'User-Agent': user_agent, 'content-type': 'application/json'}
    base_url = 'https://civitai.com/api/v1/model-versions/by-hash/'
    params = {}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    import hashlib
    with open(path, 'rb') as file:
        file_data = file.read()
    hash_sha256 = hashlib.sha256(file_data).hexdigest()
    url = base_url + hash_sha256
    try:
        r = session.get(url, params=params, headers=headers, stream=True, timeout=(3.0, 15))
    except Exception as e:
        print(e)
        return default
    else:
        if not r.ok: return None
        json = r.json()
        if 'baseModel' not in json:
            civitai_not_exists_list.append(path)
            return default
        items = []
        items.append(" / ".join(json['trainedWords']))                  # The words (prompts) used to trigger the model
        items.append(json['baseModel'])                                 # Base model (SDXL1.0, Pony, ...)
        items.append(json['model']['name'])                             # The name of the model version
        items.append(f"https://civitai.com/models/{json['modelId']}")   # The repo url for the model
        items.append(json['images'][0]['url'])                          # The url for a sample image
        loras_url_to_path_dict[path] = json['downloadUrl']              # The download url to get the model file for this specific version
        return items


def search_lora_on_civitai(query: str, allow_model: list[str] = ["Pony", "SDXL 1.0"], limit: int = 100,
                           sort: str = "Highest Rated", period: str = "AllTime", tag: str = "", user: str = "", page: int = 1):
    user_agent = get_user_agent()
    headers = {'User-Agent': user_agent, 'content-type': 'application/json'}
    if CIVITAI_API_KEY: headers['Authorization'] = f'Bearer {{{CIVITAI_API_KEY}}}'
    base_url = 'https://civitai.com/api/v1/models'
    params = {'types': ['LORA'], 'sort': sort, 'period': period, 'limit': limit, 'page': int(page), 'nsfw': 'true'}
    if query: params["query"] = query
    if tag: params["tag"] = tag
    if user: params["username"] = user
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        r = session.get(base_url, params=params, headers=headers, stream=True, timeout=(3.0, 30))
    except Exception as e:
        print(e)
        return None
    else:
        if not r.ok: return None
        json = r.json()
        if 'items' not in json: return None
        items = []
        for j in json['items']:
            for model in j['modelVersions']:
                item = {}
                if len(allow_model) != 0 and model['baseModel'] not in set(allow_model): continue
                item['name'] = j['name']
                item['creator'] = j['creator']['username'] if 'creator' in j.keys() and 'username' in j['creator'].keys() else ""
                item['tags'] = j['tags'] if 'tags' in j.keys() else []
                item['model_name'] = model['name'] if 'name' in model.keys() else ""
                item['base_model'] = model['baseModel'] if 'baseModel' in model.keys() else ""
                item['description'] = model['description'] if 'description' in model.keys() else ""
                item['dl_url'] = model['downloadUrl']
                item['md'] = ""
                if 'images' in model.keys() and len(model["images"]) != 0:
                    item['img_url'] = model["images"][0]["url"]
                    item['md'] += f'<img src="{model["images"][0]["url"]}#float" alt="thumbnail" width="150" height="240"><br>'
                else: item['img_url'] = "/home/user/app/null.png"
                item['md'] += f'''Model URL: [https://civitai.com/models/{j["id"]}](https://civitai.com/models/{j["id"]})<br>Model Name: {item["name"]}<br>
                    Creator: {item["creator"]}<br>Tags: {", ".join(item["tags"])}<br>Base Model: {item["base_model"]}<br>Description: {item["description"]}'''
                items.append(item)
        return items


def search_civitai_lora(query, base_model=[], sort=CIVITAI_SORT[0], period=CIVITAI_PERIOD[0], tag="", user="", gallery=[]):
    global civitai_last_results, civitai_last_choices, civitai_last_gallery
    civitai_last_choices = [("", "")]
    civitai_last_gallery = []
    civitai_last_results = {}
    items = search_lora_on_civitai(query, base_model, 100, sort, period, tag, user)
    if not items: return gr.update(choices=[("", "")], value="", visible=False),\
          gr.update(value="", visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    civitai_last_results = {}
    choices = []
    gallery = []
    for item in items:
        base_model_name = "Pony🐴" if item['base_model'] == "Pony" else item['base_model']
        name = f"{item['name']} (for {base_model_name} / By: {item['creator']} / Tags: {', '.join(item['tags'])})"
        value = item['dl_url']
        choices.append((name, value))
        gallery.append((item['img_url'], name))
        civitai_last_results[value] = item
    if not choices: return gr.update(choices=[("", "")], value="", visible=False),\
          gr.update(value="", visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    civitai_last_choices = choices
    civitai_last_gallery = gallery
    result = civitai_last_results.get(choices[0][1], "None")
    md = result['md'] if result else ""
    return gr.update(choices=choices, value=choices[0][1], visible=True), gr.update(value=md, visible=True),\
          gr.update(visible=True), gr.update(visible=True), gr.update(value=gallery)


def update_civitai_selection(evt: gr.SelectData):
    try:
        selected_index = evt.index
        selected = civitai_last_choices[selected_index][1]
        return gr.update(value=selected)
    except Exception:
        return gr.update()


def select_civitai_lora(search_result):
    if not "http" in search_result: return gr.update(value=""), gr.update(value="None", visible=True)
    result = civitai_last_results.get(search_result, "None")
    md = result['md'] if result else ""
    return gr.update(value=search_result), gr.update(value=md, visible=True)


def download_my_lora_flux(dl_urls: str, lora):
    path = download_lora(dl_urls)
    if path: lora = path
    choices = get_all_lora_tupled_list()
    return gr.update(value=lora, choices=choices)


def apply_lora_prompt_flux(lora_info: str):
    if lora_info == "None": return ""
    lora_tag = lora_info.replace("/",",")
    lora_tags = lora_tag.split(",") if str(lora_info) != "None" else []
    lora_prompts = normalize_prompt_list(lora_tags)
    prompt = ", ".join(list_uniq(lora_prompts))
    return prompt


def update_loras_flux(prompt, lora, lora_wt):
    on, label, tag, md = get_lora_info(lora)
    choices = get_all_lora_tupled_list()
    return gr.update(value=prompt), gr.update(value=lora, choices=choices), gr.update(value=lora_wt),\
     gr.update(value=tag, label=label, visible=on), gr.update(value=md, visible=on)


def search_civitai_lora_json(query, base_model):
    results = {}
    items = search_lora_on_civitai(query, base_model)
    if not items: return gr.update(value=results)
    for item in items:
        results[item['dl_url']] = item
    return gr.update(value=results)


def get_civitai_tag():
    default = [""]
    user_agent = get_user_agent()
    headers = {'User-Agent': user_agent, 'content-type': 'application/json'}
    base_url = 'https://civitai.com/api/v1/tags'
    params = {'limit': 200}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    url = base_url
    try:
        r = session.get(url, params=params, headers=headers, stream=True, timeout=(3.0, 15))
        if not r.ok: return default
        j = dict(r.json()).copy()
        if "items" not in j.keys(): return default
        items = []
        for item in j["items"]:
            items.append([str(item.get("name", "")), int(item.get("modelCount", 0))])
        df = pd.DataFrame(items)
        df.sort_values(1, ascending=False)
        tags = df.values.tolist()
        tags = [""] + [l[0] for l in tags]
        return tags
    except Exception as e:
        print(e)
        return default


LORA_BASE_MODEL_DICT = {
    "diffusers:StableDiffusionPipeline": ["SD 1.5"],
    "diffusers:StableDiffusionXLPipeline": ["Pony", "SDXL 1.0"],
    "diffusers:FluxPipeline": ["Flux.1 D", "Flux.1 S"],
}


def get_lora_base_model(model_name: str):
    api = HfApi(token=HF_TOKEN)
    default = ["Pony", "SDXL 1.0"]
    try:
        model = api.model_info(repo_id=model_name, timeout=5.0)
        tags = model.tags
        for tag in tags:
            if tag in LORA_BASE_MODEL_DICT.keys(): return LORA_BASE_MODEL_DICT.get(tag, default)
    except Exception:
        return default
    return default


def find_similar_lora(q: str, model_name: str):
    from rapidfuzz.process import extractOne
    from rapidfuzz.utils import default_process
    query = to_lora_key(q)
    print(f"Finding <lora:{query}:...>...")
    keys = list(private_lora_dict.keys())
    values = [x[2] for x in list(private_lora_dict.values())]
    s = default_process(query)
    e1 = extractOne(s, keys + values, processor=default_process, score_cutoff=80.0)
    key = ""
    if e1:
        e = e1[0]
        if e in set(keys): key = e
        elif e in set(values): key = keys[values.index(e)]
    if key:
        path = to_lora_path(key)
        new_path = to_lora_path(query)
        if not Path(path).exists():
            if not Path(new_path).exists(): download_private_file_from_somewhere(path, True)
            if Path(path).exists() and copy_lora(path, new_path): return new_path
    print(f"Finding <lora:{query}:...> on Civitai...")
    civitai_query = Path(query).stem if Path(query).is_file() else query
    civitai_query = civitai_query.replace("_", " ").replace("-", " ")
    base_model = get_lora_base_model(model_name)
    items = search_lora_on_civitai(civitai_query, base_model, 1)
    if items:
        item = items[0]
        path = download_lora(item['dl_url'])
        new_path = query if Path(query).is_file() else to_lora_path(query)
        if path and copy_lora(path, new_path): return new_path
    return None


def change_interface_mode(mode: str):
    if mode == "Fast":
        return gr.update(open=False), gr.update(visible=True), gr.update(open=False), gr.update(open=False),\
        gr.update(visible=True), gr.update(open=False), gr.update(visible=True), gr.update(open=False),\
        gr.update(visible=True), gr.update(value="Fast")
    elif mode == "Simple": # t2i mode
        return gr.update(open=True), gr.update(visible=True), gr.update(open=False), gr.update(open=False),\
        gr.update(visible=True), gr.update(open=False), gr.update(visible=False), gr.update(open=True),\
        gr.update(visible=False), gr.update(value="Standard")
    elif mode == "LoRA": # t2i LoRA  mode
        return gr.update(open=True), gr.update(visible=True), gr.update(open=True), gr.update(open=False),\
        gr.update(visible=True), gr.update(open=True), gr.update(visible=True), gr.update(open=False),\
        gr.update(visible=False), gr.update(value="Standard")
    else: # Standard
        return gr.update(open=False), gr.update(visible=True), gr.update(open=False), gr.update(open=False),\
        gr.update(visible=True), gr.update(open=False), gr.update(visible=True), gr.update(open=False),\
        gr.update(visible=True), gr.update(value="Standard")


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


optimization_list = {
    "None": [28, 7., 'Euler', False, 'None', 1.],
    "Default": [28, 7., 'Euler', False, 'None', 1.],
    "SPO": [28, 7., 'Euler', True, 'loras/spo_sdxl_10ep_4k-data_lora_diffusers.safetensors', 1.],
    "DPO": [28, 7., 'Euler', True, 'loras/sdxl-DPO-LoRA.safetensors', 1.],
    "DPO Turbo": [8, 2.5, 'LCM', True, 'loras/sd_xl_dpo_turbo_lora_v1-128dim.safetensors', 1.],
    "SDXL Turbo": [8, 2.5, 'LCM', True, 'loras/sd_xl_turbo_lora_v1.safetensors', 1.],
    "Hyper-SDXL 12step": [12, 5., 'TCD', True, 'loras/Hyper-SDXL-12steps-CFG-lora.safetensors', 1.],
    "Hyper-SDXL 8step": [8, 5., 'TCD', True, 'loras/Hyper-SDXL-8steps-CFG-lora.safetensors', 1.],
    "Hyper-SDXL 4step": [4, 0, 'TCD', True, 'loras/Hyper-SDXL-4steps-lora.safetensors', 1.],
    "Hyper-SDXL 2step": [2, 0, 'TCD', True, 'loras/Hyper-SDXL-2steps-lora.safetensors', 1.],
    "Hyper-SDXL 1step": [1, 0, 'TCD', True, 'loras/Hyper-SDXL-1steps-lora.safetensors', 1.],
    "PCM 16step": [16, 4., 'Euler trailing', True, 'loras/pcm_sdxl_normalcfg_16step_converted.safetensors', 1.],
    "PCM 8step": [8, 4., 'Euler trailing', True, 'loras/pcm_sdxl_normalcfg_8step_converted.safetensors', 1.],
    "PCM 4step": [4, 2., 'Euler trailing', True, 'loras/pcm_sdxl_smallcfg_4step_converted.safetensors', 1.],
    "PCM 2step": [2, 1., 'Euler trailing', True, 'loras/pcm_sdxl_smallcfg_2step_converted.safetensors', 1.],
}


def set_optimization(opt, steps_gui, cfg_gui, sampler_gui, clip_skip_gui, lora_gui, lora_scale_gui):
    if not opt in list(optimization_list.keys()): opt = "None"
    def_steps_gui = 28
    def_cfg_gui = 7.
    steps = optimization_list.get(opt, "None")[0]
    cfg = optimization_list.get(opt, "None")[1]
    sampler = optimization_list.get(opt, "None")[2]
    clip_skip = optimization_list.get(opt, "None")[3]
    lora = optimization_list.get(opt, "None")[4]
    lora_scale = optimization_list.get(opt, "None")[5]
    if opt == "None":
        steps = max(steps_gui, def_steps_gui)
        cfg = max(cfg_gui, def_cfg_gui)
        clip_skip = clip_skip_gui
    elif opt == "SPO" or opt == "DPO":
        steps = max(steps_gui, def_steps_gui)
        cfg = max(cfg_gui, def_cfg_gui)

    return gr.update(value=steps), gr.update(value=cfg), gr.update(value=sampler),\
          gr.update(value=clip_skip), gr.update(value=lora), gr.update(value=lora_scale),


# [sampler_gui, steps_gui, cfg_gui, clip_skip_gui, img_width_gui, img_height_gui, optimization_gui]
preset_sampler_setting = {
    "None": ["Euler", 28, 7., True, 1024, 1024, "None"],
    "Anime 3:4 Fast": ["LCM", 8, 2.5, True, 896, 1152, "DPO Turbo"],
    "Anime 3:4 Standard": ["Euler", 28, 7., True, 896, 1152, "None"],
    "Anime 3:4 Heavy": ["Euler", 40, 7., True, 896, 1152, "None"],
    "Anime 1:1 Fast": ["LCM", 8, 2.5, True, 1024, 1024, "DPO Turbo"],
    "Anime 1:1 Standard": ["Euler", 28, 7., True, 1024, 1024, "None"],
    "Anime 1:1 Heavy": ["Euler", 40, 7., True, 1024, 1024, "None"],
    "Photo 3:4 Fast": ["LCM", 8, 2.5, False, 896, 1152, "DPO Turbo"],
    "Photo 3:4 Standard": ["DPM++ 2M Karras", 28, 7., False, 896, 1152, "None"],
    "Photo 3:4 Heavy": ["DPM++ 2M Karras", 40, 7., False, 896, 1152, "None"],
    "Photo 1:1 Fast": ["LCM", 8, 2.5, False, 1024, 1024, "DPO Turbo"],
    "Photo 1:1 Standard": ["DPM++ 2M Karras", 28, 7., False, 1024, 1024, "None"],
    "Photo 1:1 Heavy": ["DPM++ 2M Karras", 40, 7., False, 1024, 1024, "None"],
}


def set_sampler_settings(sampler_setting):
    if not sampler_setting in list(preset_sampler_setting.keys()) or sampler_setting == "None":
        return gr.update(value="Euler"), gr.update(value=28), gr.update(value=7.), gr.update(value=True),\
              gr.update(value=1024), gr.update(value=1024), gr.update(value="None")
    v = preset_sampler_setting.get(sampler_setting, ["Euler", 28, 7., True, 1024, 1024])
    # sampler, steps, cfg, clip_skip, width, height, optimization
    return gr.update(value=v[0]), gr.update(value=v[1]), gr.update(value=v[2]), gr.update(value=v[3]),\
          gr.update(value=v[4]), gr.update(value=v[5]), gr.update(value=v[6])


preset_styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
preset_quality = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in quality_prompt_list}


def process_style_prompt(prompt: str, neg_prompt: str, styles_key: str = "None", quality_key: str = "None", type: str = "Auto"):
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

    return gr.update(value=prompt), gr.update(value=neg_prompt), gr.update(value=type) 


def set_quick_presets(genre:str = "None", type:str = "Auto", speed:str = "None", aspect:str = "None"):
    quality = "None"
    style = "None"
    sampler = "None"
    opt = "None"

    if genre == "Anime":
        if type != "None" and type != "Auto": style = "Anime"
        if aspect == "1:1":
            if speed == "Heavy":
                sampler = "Anime 1:1 Heavy"
            elif speed == "Fast":
                sampler = "Anime 1:1 Fast"
            else:
                sampler = "Anime 1:1 Standard"
        elif aspect == "3:4":
            if speed == "Heavy":
                sampler = "Anime 3:4 Heavy"
            elif speed == "Fast":
                sampler = "Anime 3:4 Fast"
            else:
                sampler = "Anime 3:4 Standard"
        if type == "Pony":
            quality = "Pony Anime Common"
        elif type == "Animagine":
            quality = "Animagine Common"
        else:
            quality = "None"
    elif genre == "Photo":
        if type != "None" and type != "Auto": style = "Photographic"
        if aspect == "1:1":
            if speed == "Heavy":
                sampler = "Photo 1:1 Heavy"
            elif speed == "Fast":
                sampler = "Photo 1:1 Fast"
            else:
                sampler = "Photo 1:1 Standard"
        elif aspect == "3:4":
            if speed == "Heavy":
                sampler = "Photo 3:4 Heavy"
            elif speed == "Fast":
                sampler = "Photo 3:4 Fast"
            else:
                sampler = "Photo 3:4 Standard"
        if type == "Pony":
            quality = "Pony Common"
        else:
            quality = "None"

    if speed == "Fast":
        opt = "DPO Turbo"
        if genre == "Anime" and type != "Pony" and type != "Auto": quality = "Animagine Light v3.1"

    return gr.update(value=quality), gr.update(value=style), gr.update(value=sampler), gr.update(value=opt), gr.update(value=type)


textual_inversion_dict = {}
try:
    with open('textual_inversion_dict.json', encoding='utf-8') as f:
        textual_inversion_dict = json.load(f)
except Exception:
    pass
textual_inversion_file_token_list = []


def get_tupled_embed_list(embed_list):
    global textual_inversion_file_list
    tupled_list = []
    for file in embed_list:
        token = textual_inversion_dict.get(Path(file).name, [Path(file).stem.replace(",",""), False])[0]
        tupled_list.append((token, file))
        textual_inversion_file_token_list.append(token)
    return tupled_list


def set_textual_inversion_prompt(textual_inversion_gui, prompt_gui, neg_prompt_gui, prompt_syntax_gui):
    ti_tags = list(textual_inversion_dict.values()) + textual_inversion_file_token_list
    tags = prompt_gui.split(",") if prompt_gui else []
    prompts = []
    for tag in tags:
        tag = str(tag).strip()
        if tag and not tag in ti_tags:
            prompts.append(tag)
    ntags = neg_prompt_gui.split(",") if neg_prompt_gui else []
    neg_prompts = []
    for tag in ntags:
        tag = str(tag).strip()
        if tag and not tag in ti_tags:
            neg_prompts.append(tag)
    ti_prompts = []
    ti_neg_prompts = []
    for ti in textual_inversion_gui:
        tokens = textual_inversion_dict.get(Path(ti).name, [Path(ti).stem.replace(",",""), False])
        is_positive = tokens[1] == True or "positive" in Path(ti).parent.name
        if is_positive: # positive prompt
            ti_prompts.append(tokens[0])
        else: # negative prompt (default)
            ti_neg_prompts.append(tokens[0])
    empty = [""]
    prompt = ", ".join(prompts + ti_prompts + empty)
    neg_prompt = ", ".join(neg_prompts + ti_neg_prompts + empty)
    return gr.update(value=prompt), gr.update(value=neg_prompt),


def get_model_pipeline(repo_id: str):
    api = HfApi(token=HF_TOKEN)
    default = "StableDiffusionPipeline"
    try:
        if not is_repo_name(repo_id): return default
        model = api.model_info(repo_id=repo_id, timeout=5.0)
    except Exception:
        return default
    if model.private or model.gated: return default
    tags = model.tags
    if not 'diffusers' in tags: return default
    if 'diffusers:FluxPipeline' in tags:
        return "FluxPipeline"
    if 'diffusers:StableDiffusionXLPipeline' in tags:
        return "StableDiffusionXLPipeline"
    elif 'diffusers:StableDiffusionPipeline' in tags:
        return "StableDiffusionPipeline"
    else:
        return default


MODEL_TYPE_KEY = {
    "model.diffusion_model.output_blocks.1.1.norm.bias": "SDXL",
    "model.diffusion_model.input_blocks.11.0.out_layers.3.weight": "SD 1.5",
    "double_blocks.0.img_attn.norm.key_norm.scale": "FLUX",
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": "FLUX",
    "model.diffusion_model.joint_blocks.9.x_block.attn.ln_k.weight": "SD 3.5",
}


def safe_clean(path: str):
    try:
        if Path(path).exists():
            if Path(path).is_dir(): shutil.rmtree(str(Path(path)))
            else: Path(path).unlink()
            print(f"Deleted: {path}")
        else: print(f"File not found: {path}")
    except Exception as e:
        print(f"Failed to delete: {path} {e}")


def read_safetensors_key(path: str):
    try:
        keys = []
        state_dict = load_file(str(Path(path)))
        for k in list(state_dict.keys()):
            keys.append(k)
            state_dict.pop(k)
    except Exception as e:
        print(e)
    finally:
        del state_dict
        torch.cuda.empty_cache()
        gc.collect()
        return keys


def get_model_type_from_key(path: str):
    default = "SDXL"
    try:
        keys = read_safetensors_key(path)
        for k, v in MODEL_TYPE_KEY.items():
            if k in set(keys):
                print(f"Model type is {v}.")
                return v
        print("Model type could not be identified.")
    except Exception:
        return default
    return default


def download_link_model(url: str, localdir: str):
    try:
        new_file = None
        new_file = get_download_file(localdir, url, CIVITAI_API_KEY)
        if not new_file or Path(new_file).suffix.lower() not in set([".safetensors", ".ckpt", ".bin", ".sft"]):
            if Path(new_file).exists(): Path(new_file).unlink()
            raise gr.Error(f"Safetensors file not found: {url}")
        model_type = get_model_type_from_key(new_file)
        return new_file, model_type
    except Exception as e:
        raise gr.Error(f"Failed to load single model file: {url} {e}")


EXAMPLES_GUI = [
    [
        "1girl, souryuu asuka langley, neon genesis evangelion, plugsuit, pilot suit, red bodysuit, sitting, crossing legs, black eye patch, cat hat, throne, symmetrical, looking down, from bottom, looking at viewer, outdoors, masterpiece, best quality, very aesthetic, absurdres",
        "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        1,
        30,
        7.5,
        True,
        -1,
        "Euler",
        1152,
        896,
        "cagliostrolab/animagine-xl-4.0",
    ],
    [
        "solo, princess Zelda OOT, score_9, score_8_up, score_8, medium breasts, cute, eyelashes, cute small face, long hair, crown braid, hairclip, pointy ears, soft curvy body, looking at viewer, smile, blush, white dress, medium body, (((holding the Master Sword))), standing, deep forest in the background",
        "score_6, score_5, score_4, busty, ugly face, mutated hands, low res, blurry face, black and white,",
        1,
        30,
        5.,
        True,
        -1,
        "Euler",
        1024,
        1024,
        "votepurchase/ponyDiffusionV6XL",
    ],
    [
        "1girl, oomuro sakurako, yuru yuri, official art, school uniform, anime artwork, anime style, studio anime, highly detailed, masterpiece, best quality, very aesthetic, absurdres",
        "photo, deformed, black and white, realism, disfigured, low contrast, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        1,
        40,
        7.0,
        True,
        -1,
        "Euler",
        1024,
        1024,
        "Raelina/Rae-Diffusion-XL-V2",
    ],
    [
        "1girl, akaza akari, yuru yuri, official art, anime screencap, anime coloring, masterpiece, best quality, absurdres",
        "bad quality, worst quality, poorly drawn, sketch, multiple views, bad anatomy, bad hands, missing fingers, extra fingers, extra digits, fewer digits, signature, watermark, username",
        1,
        28,
        5.5,
        True,
        -1,
        "Euler",
        1024,
        1024,
        "Raelina/Raehoshi-illust-XL-8",
    ],
    [
        "yoshida yuuko, machikado mazoku, 1girl, solo, demon horns,horns, school uniform, long hair, open mouth, skirt, demon girl, ahoge, shiny, shiny hair, anime artwork",
        "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        1,
        50,
        7.,
        True,
        -1,
        "Euler",
        1024,
        1024,
        "cagliostrolab/animagine-xl-4.0",
    ],
]


RESOURCES = (
    """### Resources
    - You can also try the image generator in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/SD_diffusers_interactive).
    """
)
