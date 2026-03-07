import spaces
import gradio as gr
from pathlib import Path
import re
import torch
import gc
import os
import urllib
from typing import Any
from gradio import MessageDict
from huggingface_hub import hf_hub_download, HfApi
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from ja_to_danbooru.ja_to_danbooru import jatags_to_danbooru_tags
import wrapt_timeout_decorator
from llama_cpp_agent.messages_formatter import MessagesFormatter
from formatter import mistral_v1_formatter, mistral_v2_formatter, mistral_v3_tekken_formatter
from llmenv import llm_models, llm_models_dir, llm_loras, llm_loras_dir, llm_formats, llm_languages, dolphin_system_prompt
import subprocess
subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)


llm_models_list = []
llm_loras_list = []
default_llm_model_filename = list(llm_models.keys())[0]
default_llm_lora_filename = list(llm_loras.keys())[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN", False)


def to_list(s: str):
    return [x.strip() for x in s.split(",") if not s == ""]


def list_uniq(l: list):
    return sorted(set(l), key=l.index)


DEFAULT_STATE = {
    "dolphin_sysprompt_mode": "Default",
    "dolphin_output_language": llm_languages[0],
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


@wrapt_timeout_decorator.timeout(dec_timeout=3.5)
def to_list_ja(s: str):
    s = re.sub(r'[、。]', ',', s)
    return [x.strip() for x in s.split(",") if not s == ""]


def is_japanese(s: str):
    import unicodedata
    for ch in s:
        name = unicodedata.name(ch, "") 
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


def get_dir_size(path: str):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def get_dir_size_gb(path: str):
    try:
        size_gb = get_dir_size(path) / (1024 ** 3)
        print(f"Dir size: {size_gb:.2f} GB ({path})")
    except Exception as e:
        size_gb = 999
        print(f"Error while retrieving the used storage: {e}.")
    finally:
        return size_gb


def clean_dir(path: str, size_gb: float, limit_gb: float):
    try:
        files = os.listdir(path)
        files = [os.path.join(path, f) for f in files if f.endswith(".gguf") and default_llm_model_filename not in f and default_llm_lora_filename not in f]
        files.sort(key=os.path.getatime, reverse=False)
        req_bytes = int((size_gb - limit_gb) * (1024 ** 3))
        for file in files:
            if req_bytes < 0: break
            size = os.path.getsize(file)
            Path(file).unlink()
            req_bytes -= size
            print(f"Deleted: {file}")
    except Exception as e:
        print(e)


def update_storage(path: str, limit_gb: float=50.0):
    size_gb = get_dir_size_gb(path)
    if size_gb > limit_gb:
        print("Cleaning storage...")
        clean_dir(path, size_gb, limit_gb)
        #get_dir_size_gb(path)


def split_hf_url(url: str):
    try:
        s = list(re.findall(r'^(?:https?://huggingface.co/)(?:(datasets|spaces)/)?(.+?/.+?)/\w+?/.+?/(?:(.+)/)?(.+?.\w+)(?:\?download=true)?$', url)[0])
        if len(s) < 4: return "", "", "", ""
        repo_id = s[1]
        if s[0] == "datasets": repo_type = "dataset"
        elif s[0] == "spaces": repo_type = "space"
        else: repo_type = "model"
        subfolder = urllib.parse.unquote(s[2]) if s[2] else None
        filename = urllib.parse.unquote(s[3])
        return repo_id, filename, subfolder, repo_type
    except Exception as e:
        print(e)


def hf_url_exists(url: str):
    hf_token = HF_TOKEN
    repo_id, filename, subfolder, repo_type = split_hf_url(url)
    api = HfApi(token=hf_token)
    return api.file_exists(repo_id=repo_id, filename=filename, repo_type=repo_type, token=hf_token)


def get_repo_type(repo_id: str):
    try:
        api = HfApi(token=HF_TOKEN)
        if api.repo_exists(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN): return "dataset"
        elif api.repo_exists(repo_id=repo_id, repo_type="space", token=HF_TOKEN): return "space"
        elif api.repo_exists(repo_id=repo_id, token=HF_TOKEN): return "model"
        else: return None
    except Exception as e:
        print(e)
        raise Exception(f"Repo not found: {repo_id} {e}")


def get_hf_blob_url(repo_id: str, repo_type: str, path: str):
    if repo_type == "model": return f"https://huggingface.co/{repo_id}/blob/main/{path}"
    elif repo_type == "dataset": return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path}"
    elif repo_type == "space": return f"https://huggingface.co/spaces/{repo_id}/blob/main/{path}"


def get_gguf_url(s: str):
    def find_gguf(d: dict, keys: dict):
        paths = []
        for key, size in keys.items():
            if size != 0: l = [p for p, s in d.items() if key.lower() in p.lower() and s < size]
            else: l = [p for p in d.keys() if key.lower() in p.lower()]
            if len(l) > 0: paths.append(l[0])
        if len(paths) > 0: return paths[0]
        return list(d.keys())[0]

    try:
        if s.lower().endswith(".gguf"): return s
        repo_type = get_repo_type(s)
        if repo_type is None: return s
        repo_id = s
        api = HfApi(token=HF_TOKEN)
        gguf_dict = {i.path: i.size for i in api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True, token=HF_TOKEN) if i.path.endswith(".gguf")}
        if len(gguf_dict) == 0: return s
        return get_hf_blob_url(repo_id, repo_type, find_gguf(gguf_dict, {"Q5_K_M": 6000000000, "Q4_K_M": 0, "Q4": 0}))
    except Exception as e:
        print(e)
        return s


def download_hf_file(directory, url, progress=gr.Progress(track_tqdm=True)):
    hf_token = HF_TOKEN
    repo_id, filename, subfolder, repo_type = split_hf_url(url)
    try:
        print(f"Downloading {url} to {directory}")
        if subfolder is not None: path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type=repo_type, local_dir=directory, token=hf_token)
        else: path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=directory, token=hf_token)
        return path
    except Exception as e:
        print(f"Failed to download: {e}")
        return None


def update_llm_model_list():
    global llm_models_list
    llm_models_list = []
    for k in llm_models.keys():
        llm_models_list.append(k)
    model_files = Path(llm_models_dir).glob('*.gguf')
    for path in model_files:
        llm_models_list.append(path.name)
    llm_models_list = list_uniq(llm_models_list)
    return llm_models_list


def download_llm_model(filename: str):
    if filename not in llm_models.keys(): return default_llm_model_filename
    try:
        hf_hub_download(repo_id=llm_models[filename][0], filename=filename, local_dir=llm_models_dir, token=HF_TOKEN)
    except Exception as e:
        print(e)
        return default_llm_model_filename
    update_llm_model_list()
    return filename


def update_llm_lora_list():
    global llm_loras_list
    llm_loras_list = list(llm_loras.keys()).copy()
    model_files = Path(llm_loras_dir).glob('*.gguf')
    for path in model_files:
        llm_loras_list.append(path.name)
    llm_loras_list = list_uniq([""] + llm_loras_list)
    return llm_loras_list


def download_llm_lora(filename: str):
    if not filename in llm_loras.keys(): return ""
    try:
        download_hf_file(llm_loras_dir, llm_loras[filename])
    except Exception as e:
        print(e)
        return ""
    update_llm_lora_list()
    return filename


def get_dolphin_model_info(filename: str):
    md = "None"
    items = llm_models.get(filename, None)
    if items:
        md = f'Repo: [{items[0]}](https://huggingface.co/{items[0]})'
    return md


def select_dolphin_model(filename: str, state: dict, progress=gr.Progress(track_tqdm=True)):
    set_state(state, "override_llm_format", None)
    progress(0, desc="Loading model...")
    value = download_llm_model(filename)
    progress(1, desc="Model loaded.")
    md = get_dolphin_model_info(filename)
    update_storage(llm_models_dir)
    return gr.update(value=value, choices=get_dolphin_models()), gr.update(value=get_dolphin_model_format(value)), gr.update(value=md), state


def select_dolphin_lora(filename: str, state: dict, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Loading lora...")
    value = download_llm_lora(filename)
    progress(1, desc="Lora loaded.")
    update_storage(llm_loras_dir)
    return gr.update(value=value, choices=get_dolphin_loras()), state


def select_dolphin_format(format_name: str, state: dict):
    set_state(state, "override_llm_format", llm_formats[format_name])
    return gr.update(value=format_name), state


download_llm_model(default_llm_model_filename)


def get_dolphin_models():
    return update_llm_model_list()


def get_dolphin_loras():
    return update_llm_lora_list()


def get_llm_formats():
    return list(llm_formats.keys())


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def get_dolphin_model_format(filename: str):
    if not filename in llm_models.keys(): filename = default_llm_model_filename
    format = llm_models[filename][1]
    format_name = get_key_from_value(llm_formats, format)
    return format_name


def add_dolphin_models(query: str, format_name: str):
    global llm_models
    try:
        add_models = {}
        format = llm_formats[format_name]
        filename = ""
        repo = ""
        query = get_gguf_url(query)
        if hf_url_exists(query):
            s = list(re.findall(r'^https?://huggingface.co/(.+?/.+?)/(?:blob|resolve)/main/(.+.gguf)(?:\?download=true)?$', query)[0])
            if len(s) == 2:
                repo = s[0]
                filename = s[1]
                add_models[filename] = [repo, format]
        else: return gr.update()
    except Exception as e:
        print(e)
        return gr.update()
    llm_models = (llm_models | add_models).copy()
    update_llm_model_list()
    choices = get_dolphin_models()
    return gr.update(choices=choices, value=choices[-1])


def add_dolphin_loras(query: str):
    global llm_loras
    try:
        add_loras = {}
        query = get_gguf_url(query)
        if hf_url_exists(query): add_loras[Path(query).name] = query
    except Exception as e:
        print(e)
        return gr.update()
    llm_loras = (llm_loras | add_loras).copy()
    update_llm_lora_list()
    choices = get_dolphin_loras()
    return gr.update(choices=choices, value=choices[-1])


def get_dolphin_sysprompt(state: dict={}):
    dolphin_sysprompt_mode = get_state(state, "dolphin_sysprompt_mode")
    dolphin_output_language = get_state(state, "dolphin_output_language")
    prompt = re.sub('<LANGUAGE>', dolphin_output_language if dolphin_output_language else llm_languages[0],
                    dolphin_system_prompt.get(dolphin_sysprompt_mode, dolphin_system_prompt[list(dolphin_system_prompt.keys())[0]]))
    return prompt


def get_dolphin_sysprompt_mode():
    return list(dolphin_system_prompt.keys())


def select_dolphin_sysprompt(key: str, state: dict):
    dolphin_sysprompt_mode = get_state(state, "dolphin_sysprompt_mode")
    if not key in dolphin_system_prompt.keys(): dolphin_sysprompt_mode = "Default"
    else: dolphin_sysprompt_mode = key
    set_state(state, "dolphin_sysprompt_mode", dolphin_sysprompt_mode)
    return gr.update(value=get_dolphin_sysprompt(state)), state


def get_dolphin_languages():
    return llm_languages


def select_dolphin_language(lang: str, state: dict):
    set_state(state, "dolphin_output_language", lang)
    return gr.update(value=get_dolphin_sysprompt(state)), state


@wrapt_timeout_decorator.timeout(dec_timeout=5.0)
def get_raw_prompt(msg: str):
    m = re.findall(r'/GENBEGIN/(.+?)/GENEND/', msg, re.DOTALL)
    return re.sub(r'[*/:_"#\n]', ' ', ", ".join(m)).lower() if m else ""


# https://llama-cpp-python.readthedocs.io/en/latest/api-reference/
@torch.inference_mode()
@spaces.GPU(duration=30)
def dolphin_respond(
    message: str,
    history: list[MessageDict],
    model: str = default_llm_model_filename,
    system_message: str = get_dolphin_sysprompt(),
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    lora: str = "",
    lora_scale: float = 1.0,
    state: dict = {},
    progress=gr.Progress(track_tqdm=True),
):
    try:
        model_path = Path(f"{llm_models_dir}/{model}")
        if not model_path.exists(): raise gr.Error(f"Model file not found: {str(model_path)}")
        progress(0, desc="Processing...")
        override_llm_format = get_state(state, "override_llm_format")
        if override_llm_format: chat_template = override_llm_format
        else: chat_template = llm_models[model][1]

        kwargs = {}
        if lora:
            kwargs["lora_path"] = str(Path(f"{llm_loras_dir}/{lora}"))
            kwargs["lora_scale"] = lora_scale
        else:
            kwargs["flash_attn"] = True
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=81, # 81
            n_batch=1024,
            n_ctx=8192, #8192
            **kwargs,
        )
        provider = LlamaCppPythonProvider(llm)

        agent = LlamaCppAgent(
            provider,
            system_prompt=f"{system_message}",
            predefined_messages_formatter_type=chat_template if not isinstance(chat_template, MessagesFormatter) else None,
            custom_messages_formatter=chat_template if isinstance(chat_template, MessagesFormatter) else None,
            debug_output=False
        )
        
        settings = provider.get_provider_default_settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.max_tokens = max_tokens
        settings.repeat_penalty = repeat_penalty
        settings.stream = True

        messages = BasicChatHistory()

        for msn in history:
            if msn["role"] == "user":
                user = {'role': Roles.user, 'content': msn["content"]}
                messages.add_message(user)
            elif msn["role"] == "assistant":
                assistant = {'role': Roles.assistant, 'content': msn["content"]}
                messages.add_message(assistant)
        
        stream = agent.get_chat_response(
            message,
            llm_sampling_settings=settings,
            chat_history=messages,
            returns_streaming_generator=True,
            print_output=False
        )
        
        progress(0.5, desc="Processing...")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        for output in stream:
            history[-1]['content'] += output
            yield history
    except Exception as e:
        print(e)
        raise gr.Error(f"Error: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def dolphin_parse(
    history: list[MessageDict],
    state: dict,
):
    try:
        dolphin_sysprompt_mode = get_state(state, "dolphin_sysprompt_mode")
        if dolphin_sysprompt_mode == "Chat with LLM" or not history or len(history) < 1:
            return "", gr.update(), gr.update()
        msg = history[-1]["content"]
        raw_prompt = get_raw_prompt(msg)
        prompts = []
        if dolphin_sysprompt_mode == "Japanese to Danbooru Dictionary" and is_japanese(raw_prompt):
            prompts = list_uniq(jatags_to_danbooru_tags(to_list_ja(raw_prompt)) + ["nsfw", "explicit"])
        else:
            prompts = list_uniq(to_list(raw_prompt) + ["nsfw", "explicit"])
        return ", ".join(prompts), gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        print(e)
        return "", gr.update(), gr.update()


@torch.inference_mode()
@spaces.GPU(duration=30)
def dolphin_respond_auto(
    message: str,
    history: list[MessageDict],
    model: str = default_llm_model_filename,
    system_message: str = get_dolphin_sysprompt(),
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    lora: str = "",
    lora_scale: float = 1.0,
    state: dict = {},
    progress=gr.Progress(track_tqdm=True),
):
    try:
        model_path = Path(f"{llm_models_dir}/{model}")
        #if not is_japanese(message): return [(None, None)]
        progress(0, desc="Processing...")

        override_llm_format = get_state(state, "override_llm_format")
        if override_llm_format: chat_template = override_llm_format
        else: chat_template = llm_models[model][1]

        kwargs = {}
        if lora:
            kwargs["lora_path"] = str(Path(f"{llm_loras_dir}/{lora}"))
            kwargs["lora_scale"] = lora_scale
        else:
            kwargs["flash_attn"] = True
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=81, # 81
            n_batch=1024,
            n_ctx=8192, #8192
            **kwargs,
        )
        provider = LlamaCppPythonProvider(llm)

        agent = LlamaCppAgent(
            provider,
            system_prompt=f"{system_message}",
            predefined_messages_formatter_type=chat_template if not isinstance(chat_template, MessagesFormatter) else None,
            custom_messages_formatter=chat_template if isinstance(chat_template, MessagesFormatter) else None,
            debug_output=False
        )
        
        settings = provider.get_provider_default_settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.max_tokens = max_tokens
        settings.repeat_penalty = repeat_penalty
        settings.stream = True

        messages = BasicChatHistory()

        for msn in history:
            if msn["role"] == "user":
                user = {'role': Roles.user, 'content': msn["content"]}
                messages.add_message(user)
            elif msn["role"] == "assistant":
                assistant = {'role': Roles.assistant, 'content': msn["content"]}
                messages.add_message(assistant)
        
        progress(0, desc="Translating...")
        stream = agent.get_chat_response(
            message,
            llm_sampling_settings=settings,
            chat_history=messages,
            returns_streaming_generator=True,
            print_output=False
        )

        progress(0.5, desc="Processing...")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        for output in stream:
            history[-1]['content'] += output
            yield history, gr.update(), gr.update()
    except Exception as e:
        print(e)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": message})
        yield history, gr.update(), gr.update()
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def dolphin_parse_simple(
    message: str,
    history: list[MessageDict],
    state: dict,
):
    try:
        #if not is_japanese(message): return message
        dolphin_sysprompt_mode = get_state(state, "dolphin_sysprompt_mode")
        if dolphin_sysprompt_mode == "Chat with LLM" or not history or len(history) < 1: return message
        msg = history[-1]["content"]
        raw_prompt = get_raw_prompt(msg)
        prompts = []
        if dolphin_sysprompt_mode == "Japanese to Danbooru Dictionary" and is_japanese(raw_prompt):
            prompts = list_uniq(jatags_to_danbooru_tags(to_list_ja(raw_prompt)) + ["nsfw", "explicit", "rating_explicit"])
        else:
            prompts = list_uniq(to_list(raw_prompt) + ["nsfw", "explicit", "rating_explicit"])
        return ", ".join(prompts)
    except Exception as e:
        print(e)
        return ""


# https://huggingface.co/spaces/CaioXapelaum/GGUF-Playground
import cv2
cv2.setNumThreads(1)


@torch.inference_mode()
@spaces.GPU(duration=30)
def respond_playground(
    message: str,
    history: list[MessageDict],
    model: str = default_llm_model_filename,
    system_message: str = get_dolphin_sysprompt(),
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    lora: str = "",
    lora_scale: float = 1.0,
    state: dict = {},
    progress=gr.Progress(track_tqdm=True),
):
    try:
        model_path = Path(f"{llm_models_dir}/{model}")
        if not model_path.exists(): raise gr.Error(f"Model file not found: {str(model_path)}")
        override_llm_format = get_state(state, "override_llm_format")
        if override_llm_format: chat_template = override_llm_format
        else: chat_template = llm_models[model][1]

        kwargs = {}
        if lora:
            kwargs["lora_path"] = str(Path(f"{llm_loras_dir}/{lora}"))
            kwargs["lora_scale"] = lora_scale
        else:
            kwargs["flash_attn"] = True
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=81, # 81
            n_batch=1024,
            n_ctx=8192, #8192
            **kwargs,
        )
        provider = LlamaCppPythonProvider(llm)

        agent = LlamaCppAgent(
            provider,
            system_prompt=f"{system_message}",
            predefined_messages_formatter_type=chat_template if not isinstance(chat_template, MessagesFormatter) else None,
            custom_messages_formatter=chat_template if isinstance(chat_template, MessagesFormatter) else None,
            debug_output=False
        )
        
        settings = provider.get_provider_default_settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.max_tokens = max_tokens
        settings.repeat_penalty = repeat_penalty
        settings.stream = True

        messages = BasicChatHistory()

        # Add user and assistant messages to the history
        for msn in history:
            if msn["role"] == "user":
                user = {'role': Roles.user, 'content': msn["content"]}
                messages.add_message(user)
            elif msn["role"] == "assistant":
                assistant = {'role': Roles.assistant, 'content': msn["content"]}
                messages.add_message(assistant)

        # Stream the response
        stream = agent.get_chat_response(
            message,
            llm_sampling_settings=settings,
            chat_history=messages,
            returns_streaming_generator=True,
            print_output=False
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        for output in stream:
            history[-1]['content'] += output
            yield history
    except Exception as e:
        print(e)
        raise gr.Error(f"Error: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
