import time, os
import torch
from typing import Callable
from pathlib import Path

from dartrs.v2 import (
    V2Model,
    MixtralModel,
    MistralModel,
    compose_prompt,
    LengthTag,
    AspectRatioTag,
    RatingTag,
    IdentityTag,
)
from dartrs.dartrs import DartTokenizer
from dartrs.utils import get_generation_config

import gradio as gr
from gradio.components import Component

try:
    from output import UpsamplingOutput
except:
    from .output import UpsamplingOutput

HF_TOKEN = os.getenv("HF_TOKEN", None)

V2_ALL_MODELS = {
    "dart-v2-moe-sft": {
        "repo": "p1atdev/dart-v2-moe-sft",
        "type": "sft",
        "class": MixtralModel,
    },
    "dart-v2-sft": {
        "repo": "p1atdev/dart-v2-sft",
        "type": "sft",
        "class": MistralModel,
    },
}


def prepare_models(model_config: dict):
    model_name = model_config["repo"]
    tokenizer = DartTokenizer.from_pretrained(model_name, auth_token=HF_TOKEN)
    model = model_config["class"].from_pretrained(model_name, auth_token=HF_TOKEN)

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def normalize_tags(tokenizer: DartTokenizer, tags: str):
    """Just remove unk tokens."""
    return ", ".join([tag for tag in tokenizer.tokenize(tags) if tag != "<|unk|>"])


@torch.no_grad()
def generate_tags(
    model: V2Model,
    tokenizer: DartTokenizer,
    prompt: str,
    ban_token_ids: list[int],
):
    output = model.generate(
        get_generation_config(
            prompt,
            tokenizer=tokenizer,
            temperature=1,
            top_p=0.9,
            top_k=100,
            max_new_tokens=256,
            ban_token_ids=ban_token_ids,
        ),
    )

    return output


def _people_tag(noun: str, minimum: int = 1, maximum: int = 5):
    return (
        [f"1{noun}"]
        + [f"{num}{noun}s" for num in range(minimum + 1, maximum + 1)]
        + [f"{maximum+1}+{noun}s"]
    )


PEOPLE_TAGS = (
    _people_tag("girl") + _people_tag("boy") + _people_tag("other") + ["no humans"]
)


def gen_prompt_text(output: UpsamplingOutput):
    # separate people tags (e.g. 1girl)
    people_tags = []
    other_general_tags = []
    
    for tag in output.general_tags.split(","):
        tag = tag.strip()
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        else:
            other_general_tags.append(tag)

    return ", ".join(
        [
            part.strip()
            for part in [
                *people_tags,
                output.character_tags,
                output.copyright_tags,
                *other_general_tags,
                output.upsampled_tags,
                output.rating_tag,
            ]
            if part.strip() != ""
        ]
    )


def elapsed_time_format(elapsed_time: float) -> str:
    return f"Elapsed: {elapsed_time:.2f} seconds"


def parse_upsampling_output(
    upsampler: Callable[..., UpsamplingOutput],
):
    def _parse_upsampling_output(*args) -> tuple[str, str, dict]:
        output = upsampler(*args)

        return (
            gen_prompt_text(output),
            elapsed_time_format(output.elapsed_time),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    return _parse_upsampling_output


class V2UI:
    model_name: str | None = None
    model: V2Model
    tokenizer: DartTokenizer

    input_components: list[Component] = []
    generate_btn: gr.Button

    def on_generate(
        self,
        model_name: str,
        copyright_tags: str,
        character_tags: str,
        general_tags: str,
        rating_tag: RatingTag,
        aspect_ratio_tag: AspectRatioTag,
        length_tag: LengthTag,
        identity_tag: IdentityTag,
        ban_tags: str,
        *args,
    ) -> UpsamplingOutput:
        if self.model_name is None or self.model_name != model_name:
            models = prepare_models(V2_ALL_MODELS[model_name])
            self.model = models["model"]
            self.tokenizer = models["tokenizer"]
            self.model_name = model_name

        # normalize tags
        # copyright_tags = normalize_tags(self.tokenizer, copyright_tags)
        # character_tags = normalize_tags(self.tokenizer, character_tags)
        # general_tags = normalize_tags(self.tokenizer, general_tags)

        ban_token_ids = self.tokenizer.encode(ban_tags.strip())

        prompt = compose_prompt(
            prompt=general_tags,
            copyright=copyright_tags,
            character=character_tags,
            rating=rating_tag,
            aspect_ratio=aspect_ratio_tag,
            length=length_tag,
            identity=identity_tag,
        )

        start = time.time()
        upsampled_tags = generate_tags(
            self.model,
            self.tokenizer,
            prompt,
            ban_token_ids,
        )
        elapsed_time = time.time() - start

        return UpsamplingOutput(
            upsampled_tags=upsampled_tags,
            copyright_tags=copyright_tags,
            character_tags=character_tags,
            general_tags=general_tags,
            rating_tag=rating_tag,
            aspect_ratio_tag=aspect_ratio_tag,
            length_tag=length_tag,
            identity_tag=identity_tag,
            elapsed_time=elapsed_time,
        )


def parse_upsampling_output_simple(upsampler: UpsamplingOutput):
    return gen_prompt_text(upsampler)


v2 = V2UI()


def v2_upsampling_prompt(model: str = "dart-v2-moe-sft", copyright: str = "", character: str = "",
                          general_tags: str = "", rating: str = "nsfw", aspect_ratio: str = "square",
                            length: str = "very_long", identity: str = "lax", ban_tags: str = "censored"):
    raw_prompt = parse_upsampling_output_simple(v2.on_generate(model, copyright, character, general_tags,
                                                                rating, aspect_ratio, length, identity, ban_tags))
    return raw_prompt


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


def select_random_character(series: str, character: str):
    from random import seed, randrange
    seed()
    character_list = list(anime_series_dict.keys())
    character = character_list[randrange(len(character_list) - 1)]
    series = anime_series_dict.get(character.split(",")[0].strip(), "")
    return series, character


def v2_random_prompt(general_tags: str = "", copyright: str = "", character: str = "", rating: str = "nsfw",
                      aspect_ratio: str = "square", length: str = "very_long", identity: str = "lax",
                      ban_tags: str = "censored", model: str = "dart-v2-moe-sft"):
    if copyright == "" and character == "":
        copyright, character = select_random_character("", "")
    raw_prompt = v2_upsampling_prompt(model, copyright, character, general_tags, rating,
                                       aspect_ratio, length, identity, ban_tags)
    return raw_prompt, copyright, character