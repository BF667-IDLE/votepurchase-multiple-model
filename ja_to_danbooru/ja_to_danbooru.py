import argparse
import re
from pathlib import Path


def load_json_dict(path: str):
    import json
    from pathlib import Path
    dict = {}
    if not Path(path).exists(): return dict
    try:
        with open(path, encoding='utf-8') as f:
            dict = json.load(f)
    except Exception:
        print(f"Failed to open dictionary file: {path}")
        return dict
    return dict


ja_danbooru_dict = load_json_dict('ja_danbooru_dict.json')
char_series_dict = load_json_dict('character_series_dict.json')
tagtype_dict = load_json_dict('danbooru_tagtype_dict.json')


def jatags_to_danbooru_tags(jatags: list[str]):
    from rapidfuzz.process import extractOne
    from rapidfuzz.utils import default_process
    keys = list(ja_danbooru_dict.keys())
    ckeys = list(char_series_dict.keys())
    tags = []
    for jatag in jatags:
        jatag = str(jatag).strip()
        s = default_process(str(jatag))
        e1 = extractOne(s, keys, processor=default_process, score_cutoff=90.0)
        if e1:
            tag = str(ja_danbooru_dict[e1[0]])
            tags.append(tag)
            if tag in tagtype_dict.keys() and tagtype_dict[tag] == "character":
                cs = default_process(tag)
                ce1 = extractOne(cs, ckeys, processor=default_process, score_cutoff=95.0)
                if ce1:
                    series = str(char_series_dict[ce1[0]])
                    tags.append(series)
    return tags


def jatags_to_danbooru(input_tag, input_file, output_file, is_append):
    if input_file and Path(input_file).exists():
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_tag = f.read()
        except Exception:
            print(f"Failed to open input file: {input_file}")
    ja_tags = [tag.strip() for tag in input_tag.split(",")] if input_tag else []
    tags = jatags_to_danbooru_tags(ja_tags)
    output_tags = ja_tags + tags if is_append else tags
    output_tag = ", ".join(output_tags)
    if output_file:
        try:
            with open(output_file, mode='w', encoding="utf-8") as f:
                f.write(output_tag)
        except Exception:
            print(f"Failed to write output file: {output_file}")
    else:
        print(output_tag)
    return output_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default=None, type=str, required=False, help="Input tags.")
    parser.add_argument("--file", default=None, type=str, required=False, help="Input tags from a text file.")
    parser.add_argument("--out", default=None, type=str, help="Output to text file.")
    parser.add_argument("--append", default=False, type=bool, help="Whether the output contains the input tags or not.")

    args = parser.parse_args()
    assert (args.tags, args.file) != (None, None), "Must provide --tags or --file!"

    jatags_to_danbooru(args.tags, args.file, args.out, args.append)


# Usage:
# python ja_to_danbooru.py --tags "女の子, 大室櫻子"
# python danbooru_to_ja.py --file inputtag.txt
# python danbooru_to_ja.py --file inputtag.txt --append True
# Datasets: https://huggingface.co/datasets/p1atdev/danbooru-ja-tag-pair-20240715
# Datasets: https://github.com/ponapon280/danbooru-e621-converter