import spaces
from transformers import AutoProcessor, AutoModelForCausalLM
import re
from PIL import Image 
import torch

from transformers.utils import is_flash_attn_2_available
if not is_flash_attn_2_available():
    import subprocess
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    fl_model = AutoModelForCausalLM.from_pretrained('gokaygokay/Florence-2-SD3-Captioner', trust_remote_code=True).to("cpu").eval()
    fl_processor = AutoProcessor.from_pretrained('gokaygokay/Florence-2-SD3-Captioner', trust_remote_code=True)
except Exception as e:
    print(e)
    fl_model = fl_processor = None

def fl_modify_caption(caption: str) -> str:
    """
    Removes specific prefixes from captions if present, otherwise returns the original caption.
    Args:
        caption (str): A string containing a caption.
    Returns:
        str: The caption with the prefix removed if it was present, or the original caption.
    """
    # Define the prefixes to remove
    prefix_substrings = [
        ('captured from ', ''),
        ('captured at ', '')
    ]
    
    # Create a regex pattern to match any of the prefixes
    pattern = '|'.join([re.escape(opening) for opening, _ in prefix_substrings])
    replacers = {opening.lower(): replacer for opening, replacer in prefix_substrings}
    
    # Function to replace matched prefix with its corresponding replacement
    def replace_fn(match):
        return replacers[match.group(0).lower()]
    
    # Apply the regex to the caption
    modified_caption = re.sub(pattern, replace_fn, caption, count=1, flags=re.IGNORECASE)
    
    # If the caption was modified, return the modified version; otherwise, return the original
    return modified_caption if modified_caption != caption else caption


@spaces.GPU(duration=10)
def fl_run_example(image):
    task_prompt = "<DESCRIPTION>"
    prompt = task_prompt + "Describe this image in great detail."

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    fl_model.to(device)
    inputs = fl_processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = fl_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    fl_model.to("cpu")
    generated_text = fl_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = fl_processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return fl_modify_caption(parsed_answer["<DESCRIPTION>"])


def predict_tags_fl2_sd3(image: Image.Image, input_tags: str, algo: list[str]):
    def to_list(s):
        return [x.strip() for x in s.split(",") if not s == ""]
    
    def list_uniq(l):
        return sorted(set(l), key=l.index)
    
    if not "Use Florence-2-SD3-Long-Captioner" in algo:
        return input_tags
    tag_list = list_uniq(to_list(input_tags) + to_list(fl_run_example(image) + ", "))
    tag_list.remove("")
    return ", ".join(tag_list)
