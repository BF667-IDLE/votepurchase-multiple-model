import spaces
from PIL import Image
import torch
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path


WD_MODEL_NAMES = ["p1atdev/wd-swinv2-tagger-v3-hf"]
WD_MODEL_NAME = WD_MODEL_NAMES[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
default_device = device

try:
    wd_model = AutoModelForImageClassification.from_pretrained(WD_MODEL_NAME, trust_remote_code=True).to(default_device).eval()
    wd_processor = AutoImageProcessor.from_pretrained(WD_MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(e)
    wd_model = wd_processor = None

def _people_tag(noun: str, minimum: int = 1, maximum: int = 5):
    return (
        [f"1{noun}"]
        + [f"{num}{noun}s" for num in range(minimum + 1, maximum + 1)]
        + [f"{maximum+1}+{noun}s"]
    )


PEOPLE_TAGS = (
    _people_tag("girl") + _people_tag("boy") + _people_tag("other") + ["no humans"]
)


RATING_MAP = {
    "sfw": "safe",
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit, nsfw",
}
DANBOORU_TO_E621_RATING_MAP = {
    "sfw": "rating_safe",
    "general": "rating_safe",
    "safe": "rating_safe",
    "sensitive": "rating_safe",
    "nsfw": "rating_explicit",
    "explicit, nsfw": "rating_explicit",
    "explicit": "rating_explicit",
    "rating:safe": "rating_safe",
    "rating:general": "rating_safe",
    "rating:sensitive": "rating_safe",
    "rating:questionable, nsfw": "rating_explicit",
    "rating:explicit, nsfw": "rating_explicit",
}


# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def replace_underline(x: str):
    return x.strip().replace("_", " ") if x not in kaomojis else x.strip()


def to_list(s):
    return [x.strip() for x in s.split(",") if not s == ""]


def list_sub(a, b):
    return [e for e in a if e not in b]


def list_uniq(l):
    return sorted(set(l), key=l.index)


def load_dict_from_csv(filename):
    dict = {}
    if not Path(filename).exists():
        if Path('./tagger/', filename).exists(): filename = str(Path('./tagger/', filename))
        else: return dict
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        print(f"Failed to open dictionary file: {filename}")
        return dict
    for line in lines:
        parts = line.strip().split(',')
        dict[parts[0]] = parts[1]
    return dict


anime_series_dict = load_dict_from_csv('character_series_dict.csv')


def character_list_to_series_list(character_list):
    output_series_tag = []
    series_tag = ""
    series_dict = anime_series_dict
    for tag in character_list:
        series_tag = series_dict.get(tag, "")
        if tag.endswith(")"):
            tags = tag.split("(")
            character_tag = "(".join(tags[:-1])
            if character_tag.endswith(" "):
                character_tag = character_tag[:-1]
            series_tag = tags[-1].replace(")", "")

    if series_tag:
        output_series_tag.append(series_tag)

    return output_series_tag


def select_random_character(series: str, character: str):
    from random import seed, randrange
    seed()
    character_list = list(anime_series_dict.keys())
    character = character_list[randrange(len(character_list) - 1)]
    series = anime_series_dict.get(character.split(",")[0].strip(), "")
    return series, character


def danbooru_to_e621(dtag, e621_dict):
    def d_to_e(match, e621_dict):
        dtag = match.group(0)
        etag = e621_dict.get(replace_underline(dtag), "")
        if etag:
            return etag
        else:
            return dtag
    
    import re
    tag = re.sub(r'[\w ]+', lambda wrapper: d_to_e(wrapper, e621_dict), dtag, 2)
    return tag


danbooru_to_e621_dict = load_dict_from_csv('danbooru_e621.csv')


def convert_danbooru_to_e621_prompt(input_prompt: str = "", prompt_type: str = "danbooru"):
    if prompt_type == "danbooru": return input_prompt
    tags = input_prompt.split(",") if input_prompt else []
    people_tags: list[str] = []
    other_tags: list[str] = []
    rating_tags: list[str] = []

    e621_dict = danbooru_to_e621_dict
    for tag in tags:
        tag = replace_underline(tag)
        tag = danbooru_to_e621(tag, e621_dict)
        if tag in PEOPLE_TAGS:        
            people_tags.append(tag)
        elif tag in DANBOORU_TO_E621_RATING_MAP.keys():
            rating_tags.append(DANBOORU_TO_E621_RATING_MAP.get(tag.replace(" ",""), ""))            
        else:
            other_tags.append(tag)

    rating_tags = sorted(set(rating_tags), key=rating_tags.index)
    rating_tags = [rating_tags[0]] if rating_tags else []
    rating_tags = ["explicit, nsfw"] if rating_tags and rating_tags[0] == "explicit" else rating_tags

    output_prompt = ", ".join(people_tags + other_tags + rating_tags)
    
    return output_prompt


from translatepy import Translator
translator = Translator()
def translate_prompt_old(prompt: str = ""):
    def translate_to_english(input: str):
        try:
            output = str(translator.translate(input, 'English'))
        except Exception as e:
            output = input
            print(e)
        return output

    def is_japanese(s):
        import unicodedata
        for ch in s:
            name = unicodedata.name(ch, "") 
            if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
                return True
        return False

    def to_list(s):
        return [x.strip() for x in s.split(",")]
    
    prompts = to_list(prompt)
    outputs = []
    for p in prompts:
        p = translate_to_english(p) if is_japanese(p) else p
        outputs.append(p)

    return ", ".join(outputs)


def translate_prompt(input: str):
    try:
        output = str(translator.translate(input, 'English'))
    except Exception as e:
        output = input
        print(e)
    return output


def translate_prompt_to_ja(prompt: str = ""):
    def translate_to_japanese(input: str):
        try:
            output = str(translator.translate(input, 'Japanese'))
        except Exception as e:
            output = input
            print(e)
        return output

    def is_japanese(s):
        import unicodedata
        for ch in s:
            name = unicodedata.name(ch, "") 
            if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
                return True
        return False

    def to_list(s):
        return [x.strip() for x in s.split(",")]
    
    prompts = to_list(prompt)
    outputs = []
    for p in prompts:
        p = translate_to_japanese(p) if not is_japanese(p) else p
        outputs.append(p)

    return ", ".join(outputs)


def tags_to_ja(itag, dict):
    def t_to_j(match, dict):
        tag = match.group(0)
        ja = dict.get(replace_underline(tag), "")
        if ja:
            return ja
        else:
            return tag
    
    import re
    tag = re.sub(r'[\w ]+', lambda wrapper: t_to_j(wrapper, dict), itag, 2)

    return tag


def convert_tags_to_ja(input_prompt: str = ""):
    tags = input_prompt.split(",") if input_prompt else []
    out_tags = []

    tags_to_ja_dict = load_dict_from_csv('all_tags_ja_ext.csv')
    dict = tags_to_ja_dict
    for tag in tags:
        tag = replace_underline(tag)
        tag = tags_to_ja(tag, dict)
        out_tags.append(tag)
    
    return ", ".join(out_tags)


animagine_ps = to_list("masterpiece, best quality, very aesthetic, absurdres")
animagine_nps = to_list("lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]")
pony_ps = to_list("score_9, score_8_up, score_7_up, masterpiece, best quality, very aesthetic, absurdres")
pony_nps = to_list("source_pony, score_6, score_5, score_4, busty, ugly face, mutated hands, low res, blurry face, black and white, the simpsons, overwatch, apex legends")
other_ps = to_list("anime artwork, anime style, studio anime, highly detailed, cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed")
other_nps = to_list("photo, deformed, black and white, realism, disfigured, low contrast, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly")
default_ps = to_list("highly detailed, masterpiece, best quality, very aesthetic, absurdres")
default_nps = to_list("score_6, score_5, score_4, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]")
def insert_recom_prompt(prompt: str = "", neg_prompt: str = "", type: str = "None"):
    prompts = to_list(prompt)
    neg_prompts = to_list(neg_prompt)

    prompts = list_sub(prompts, animagine_ps + pony_ps)
    neg_prompts = list_sub(neg_prompts, animagine_nps + pony_nps)

    last_empty_p = [""] if not prompts and type != "None" else []
    last_empty_np = [""] if not neg_prompts and type != "None" else []

    if type == "Animagine":
        prompts = prompts + animagine_ps
        neg_prompts = neg_prompts + animagine_nps
    elif type == "Pony":
        prompts = prompts + pony_ps
        neg_prompts = neg_prompts + pony_nps

    prompt = ", ".join(list_uniq(prompts) + last_empty_p)
    neg_prompt = ", ".join(list_uniq(neg_prompts) + last_empty_np)

    return prompt, neg_prompt


def load_model_prompt_dict():
    import json
    dict = {}
    path = 'model_dict.json' if Path('model_dict.json').exists() else './tagger/model_dict.json'
    try:
        with open(path, encoding='utf-8') as f:
            dict = json.load(f)
    except Exception:
        pass
    return dict


model_prompt_dict = load_model_prompt_dict()


def insert_model_recom_prompt(prompt: str = "", neg_prompt: str = "", model_name: str = "None", type = "Auto"):
    enable_auto_recom_prompt = True if type == "Auto" else False
    if not model_name or not enable_auto_recom_prompt: return prompt, neg_prompt
    prompts = to_list(prompt)
    neg_prompts = to_list(neg_prompt)
    prompts = list_sub(prompts, animagine_ps + pony_ps + other_ps)
    neg_prompts = list_sub(neg_prompts, animagine_nps + pony_nps + other_nps)
    last_empty_p = [""] if not prompts and type != "None" else []
    last_empty_np = [""] if not neg_prompts and type != "None" else []
    ps = []
    nps = []
    if model_name in model_prompt_dict.keys(): 
        ps = to_list(model_prompt_dict[model_name]["prompt"])
        nps = to_list(model_prompt_dict[model_name]["negative_prompt"])
    else:
        ps = default_ps
        nps = default_nps
    prompts = prompts + ps
    neg_prompts = neg_prompts + nps
    prompt = ", ".join(list_uniq(prompts) + last_empty_p)
    neg_prompt = ", ".join(list_uniq(neg_prompts) + last_empty_np)
    return prompt, neg_prompt


tag_group_dict = load_dict_from_csv('tag_group.csv')


def remove_specific_prompt(input_prompt: str = "", keep_tags: str = "all"):
    def is_dressed(tag):
        import re
        p = re.compile(r'dress|cloth|uniform|costume|vest|sweater|coat|shirt|jacket|blazer|apron|leotard|hood|sleeve|skirt|shorts|pant|loafer|ribbon|necktie|bow|collar|glove|sock|shoe|boots|wear|emblem')
        return p.search(tag)

    def is_background(tag):
        import re
        p = re.compile(r'background|outline|light|sky|build|day|screen|tree|city')
        return p.search(tag)

    un_tags = ['solo']
    group_list = ['groups', 'body_parts', 'attire', 'posture', 'objects', 'creatures', 'locations', 'disambiguation_pages', 'commonly_misused_tags', 'phrases', 'verbs_and_gerunds', 'subjective', 'nudity', 'sex_objects', 'sex', 'sex_acts', 'image_composition', 'artistic_license', 'text', 'year_tags', 'metatags']
    keep_group_dict = {
        "body": ['groups', 'body_parts'],
        "dress": ['groups', 'body_parts', 'attire'],
        "all": group_list,
    }

    def is_necessary(tag, keep_tags, group_dict):
        if keep_tags == "all":
            return True
        elif tag in un_tags or group_dict.get(tag, "") in explicit_group:
            return False
        elif keep_tags == "body" and is_dressed(tag):
            return False
        elif is_background(tag):
            return False
        else:
            return True
    
    if keep_tags == "all": return input_prompt
    keep_group = keep_group_dict.get(keep_tags, keep_group_dict["body"])
    explicit_group = list(set(group_list) ^ set(keep_group))

    tags = input_prompt.split(",") if input_prompt else []
    people_tags: list[str] = []
    other_tags: list[str] = []

    group_dict = tag_group_dict
    for tag in tags:
        tag = replace_underline(tag)
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        elif is_necessary(tag, keep_tags, group_dict):
            other_tags.append(tag)

    output_prompt = ", ".join(people_tags + other_tags)
    
    return output_prompt


def sort_taglist(tags: list[str]):
    if not tags: return []
    character_tags: list[str] = []
    series_tags: list[str] = []
    people_tags: list[str] = []
    group_list = ['groups', 'body_parts', 'attire', 'posture', 'objects', 'creatures', 'locations', 'disambiguation_pages', 'commonly_misused_tags', 'phrases', 'verbs_and_gerunds', 'subjective', 'nudity', 'sex_objects', 'sex', 'sex_acts', 'image_composition', 'artistic_license', 'text', 'year_tags', 'metatags']
    group_tags = {}
    other_tags: list[str] = []
    rating_tags: list[str] = []

    group_dict = tag_group_dict
    group_set = set(group_dict.keys())
    character_set = set(anime_series_dict.keys())
    series_set = set(anime_series_dict.values())
    rating_set = set(DANBOORU_TO_E621_RATING_MAP.keys()) | set(DANBOORU_TO_E621_RATING_MAP.values())

    for tag in tags:
        tag = replace_underline(tag)
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        elif tag in rating_set:
            rating_tags.append(tag)
        elif tag in group_set:
            elem = group_dict[tag]
            group_tags[elem] = group_tags[elem] + [tag] if elem in group_tags else [tag]
        elif tag in character_set:
            character_tags.append(tag)
        elif tag in series_set:
            series_tags.append(tag)
        else:
            other_tags.append(tag)

    output_group_tags: list[str] = []
    for k in group_list:
        output_group_tags.extend(group_tags.get(k, []))

    rating_tags = [rating_tags[0]] if rating_tags else []
    rating_tags = ["explicit, nsfw"] if rating_tags and rating_tags[0] == "explicit" else rating_tags

    output_tags = character_tags + series_tags + people_tags + output_group_tags + other_tags + rating_tags
    
    return output_tags


def sort_tags(tags: str):
    if not tags: return ""
    taglist: list[str] = []
    for tag in tags.split(","):
        taglist.append(tag.strip())
    taglist = list(filter(lambda x: x != "", taglist))
    return ", ".join(sort_taglist(taglist))


def postprocess_results(results: dict[str, float], general_threshold: float, character_threshold: float):
    results = {
        k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)
    }

    rating = {}
    character = {}
    general = {}

    for k, v in results.items():
        if k.startswith("rating:"):
            rating[k.replace("rating:", "")] = v
            continue
        elif k.startswith("character:"):
            character[k.replace("character:", "")] = v
            continue

        general[k] = v

    character = {k: v for k, v in character.items() if v >= character_threshold}
    general = {k: v for k, v in general.items() if v >= general_threshold}

    return rating, character, general


def gen_prompt(rating: list[str], character: list[str], general: list[str]):
    people_tags: list[str] = []
    other_tags: list[str] = []
    rating_tag = RATING_MAP[rating[0]]

    for tag in general:
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        else:
            other_tags.append(tag)

    all_tags = people_tags + other_tags

    return ", ".join(all_tags)


@spaces.GPU(duration=10)
def predict_tags(image: Image.Image, general_threshold: float = 0.3, character_threshold: float = 0.8):
    inputs = wd_processor.preprocess(image, return_tensors="pt")

    outputs = wd_model(**inputs.to(wd_model.device, wd_model.dtype))
    logits = torch.sigmoid(outputs.logits[0])  # take the first logits

    # get probabilities
    if device != default_device: wd_model.to(device=device)
    results = {
        wd_model.config.id2label[i]: float(logit.float()) for i, logit in enumerate(logits)
    }
    if device != default_device: wd_model.to(device=default_device)
    # rating, character, general
    rating, character, general = postprocess_results(
        results, general_threshold, character_threshold
    )
    prompt = gen_prompt(
        list(rating.keys()), list(character.keys()), list(general.keys())
    )
    output_series_tag = ""
    output_series_list = character_list_to_series_list(character.keys())
    if output_series_list:
        output_series_tag = output_series_list[0]
    else:
        output_series_tag = ""
    return output_series_tag, ", ".join(character.keys()), prompt, gr.update(interactive=True)


def predict_tags_wd(image: Image.Image, input_tags: str, algo: list[str], general_threshold: float = 0.3,
                     character_threshold: float = 0.8, input_series: str = "", input_character: str = ""):
    if not "Use WD Tagger" in algo and len(algo) != 0:
        return input_series, input_character, input_tags, gr.update(interactive=True)
    return predict_tags(image, general_threshold, character_threshold)


def compose_prompt_to_copy(character: str, series: str, general: str):
    characters = character.split(",") if character else []
    serieses = series.split(",") if series else []
    generals = general.split(",") if general else []
    tags = characters + serieses + generals
    cprompt = ",".join(tags) if tags else ""
    return cprompt
