import os
from constants import LOAD_DIFFUSERS_FORMAT_MODEL as LOAD_DIFFUSERS_FORMAT_MODEL_DC

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_READ_TOKEN = os.environ.get('HF_READ_TOKEN') # only use for private repo

# - **List Models**
LOAD_DIFFUSERS_FORMAT_MODEL = [
    'cagliostrolab/animagine-xl-4.0',
    'cagliostrolab/animagine-xl-4.0-zero',
    'votepurchase/animagine-xl-3.1',
    'votepurchase/NSFW-GEN-ANIME-v2',
    'votepurchase/kivotos-xl-2.0',
    'votepurchase/holodayo-xl-2.1',
    'votepurchase/ponyDiffusionV6XL',
    'votepurchase/AnythingXL_xl',
    'votepurchase/7thAnimeXLPonyA_v10',
    'votepurchase/ChilloutMix',
    'votepurchase/NovelAIRemix',
    'votepurchase/NSFW-gen-v2',
    'votepurchase/PerfectDeliberate-Anime_v2',
    'votepurchase/realpony-xl',
    'votepurchase/artiwaifu-diffusion-1.0',
    'votepurchase/Starry-XL-v5.2',
    'votepurchase/Yaki-Dofu-Mix',
    'votepurchase/ebara-pony-v1-sdxl',
    'votepurchase/waiANIMIXPONYXL_v10',
    'votepurchase/counterfeitV30_v30',
    'votepurchase/ebara-pony',
    'votepurchase/Realistic_Vision_V1.4',
    'votepurchase/pony',
    'votepurchase/ponymatureSDXL_ponyeclipse10',
    'votepurchase/waiREALMIX_v70',
    'votepurchase/waiREALCN_v10',
    'votepurchase/PVCStyleModelMovable_pony151',
    'votepurchase/PVCStyleModelMovable_beta27Realistic',
    'votepurchase/PVCStyleModelFantasy_beta12',
    'votepurchase/pvcxl-v1-lora',
    'votepurchase/Realistic_Vision_V2.0',
    'votepurchase/RealVisXL_V4.0',
    'votepurchase/juggernautXL_hyper_8step_sfw',
    'votepurchase/ponyRealism_v21MainVAE',
    'KBlueLeaf/kohaku-xl-beta7.1',
    'KBlueLeaf/Kohaku-XL-Epsilon-rev2',
    'KBlueLeaf/Kohaku-XL-Epsilon-rev3',
    'KBlueLeaf/Kohaku-XL-Zeta',
    'kayfahaarukku/UrangDiffusion-2.0',
    'kayfahaarukku/irAsu-1.0',
    'Eugeoter/artiwaifu-diffusion-2.0',
    'comin/IterComp',
    'Emanon14/NONAMEmix_v1',
    'Spestly/OdysseyXL-4.0',
    'Spestly/OdysseyXL-MK2-Alpha',
    'Spestly/OdysseyXL-MK2-Beta',
    'hanzogak/comradeshipXL-v14T14H',
    'hanzogak/comradeshipXL-v14VT',
    '6DammK9/AstolfoKarmix-XL',
    'motimalu/kirazuri-lazuli-noobai-xl-vpred-2.0',
    'BlueDancer/Artisanica_XL',
    'neta-art/neta-noob-1.0',
    'OnomaAIResearch/Illustrious-xl-early-release-v0',
    'Laxhar/noobai-XL-1.0',
    'Laxhar/noobai-XL-Vpred-1.0',
    'Raelina/Rae-Diffusion-XL-V2',
    'Raelina/Raemu-XL-V4',
    'Raelina/Raemu-XL-V5',
    'Raelina/Raena-XL-V2',
    'Raelina/Raehoshi-illust-XL',
    'Raelina/Raehoshi-illust-xl-2',
    'Raelina/Raehoshi-Illust-XL-2.1',
    'Raelina/Raehoshi-illust-XL-3',
    'Raelina/Raehoshi-illust-XL-4',
    'Raelina/Raehoshi-illust-XL-5',
    'Raelina/Raehoshi-illust-XL-5.1',
    'Raelina/Raehoshi-illust-XL-6',
    'Raelina/Raehoshi-illust-XL-7',
    'Raelina/Raehoshi-illust-XL-7.1',
    'Raelina/Raehoshi-illust-XL-8',
    'camenduru/FLUX.1-dev-diffusers',
    'black-forest-labs/FLUX.1-schnell',
    'sayakpaul/FLUX.1-merged',
    'ostris/OpenFLUX.1',
    'multimodalart/FLUX.1-dev2pro-full',
    'Raelina/Raemu-Flux',
]
LOAD_DIFFUSERS_FORMAT_MODEL = LOAD_DIFFUSERS_FORMAT_MODEL + LOAD_DIFFUSERS_FORMAT_MODEL_DC

DIFFUSERS_FORMAT_LORAS = [
    "nerijs/animation2k-flux",
    "XLabs-AI/flux-RealismLora",
    "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
]

# List all Models for specified user
HF_MODEL_USER_LIKES = ["votepurchase"] # sorted by number of likes
HF_MODEL_USER_EX = ["John6666"] # sorted by a special rule

# - **Download Models**
DOWNLOAD_MODEL_LIST = [
]

# - **Download VAEs**
DOWNLOAD_VAE_LIST = [
    'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors?download=true',
    'https://huggingface.co/nubby/blessed-sdxl-vae-fp16-fix/resolve/main/sdxl_vae-fp16fix-c-1.1-b-0.5.safetensors?download=true',
    "https://huggingface.co/nubby/blessed-sdxl-vae-fp16-fix/blob/main/sdxl_vae-fp16fix-blessed.safetensors",
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
    "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt",
]

# - **Download LoRAs**
DOWNLOAD_LORA_LIST = [
]

# Download Embeddings
DOWNLOAD_EMBEDS = [
    'https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt_version2.pt',
    'https://huggingface.co/embed/negative/resolve/main/EasyNegativeV2.safetensors',
    'https://huggingface.co/embed/negative/resolve/main/bad-hands-5.pt',
]

DIRECTORY_MODELS = 'models'
DIRECTORY_LORAS = 'loras'
DIRECTORY_VAES = 'vaes'
DIRECTORY_EMBEDS = 'embedings'
DIRECTORY_EMBEDS_SDXL = 'embedings_xl'
DIRECTORY_EMBEDS_POSITIVE_SDXL = 'embedings_xl/positive'

directories = [DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS, DIRECTORY_EMBEDS_SDXL, DIRECTORY_EMBEDS_POSITIVE_SDXL]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

HF_LORA_PRIVATE_REPOS1 = ['John6666/loraflux1', 'John6666/loratest1', 'John6666/loratest3', 'John6666/loratest4', 'John6666/loratest6']
HF_LORA_PRIVATE_REPOS2 = ['John6666/loratest10', 'John6666/loratest11','John6666/loratest'] # to be sorted as 1 repo
HF_LORA_PRIVATE_REPOS = HF_LORA_PRIVATE_REPOS1 + HF_LORA_PRIVATE_REPOS2
HF_LORA_ESSENTIAL_PRIVATE_REPO = 'John6666/loratest1' # to be downloaded on run app
HF_VAE_PRIVATE_REPO = 'John6666/vaetest'
HF_SDXL_EMBEDS_NEGATIVE_PRIVATE_REPO = 'John6666/embeddingstest'
HF_SDXL_EMBEDS_POSITIVE_PRIVATE_REPO = 'John6666/embeddingspositivetest'
