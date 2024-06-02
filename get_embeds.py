# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00000-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00001-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00002-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00003-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00004-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00005-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00006-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00007-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00008-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet
# !wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00009-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet

import transformers
import torch
import glob
import pandas as pd
from tqdm import tqdm

paths = glob.glob("./*.parquet")
dfs = [pd.read_parquet(path) for path in paths]
df = pd.concat(dfs)
df = df[df.aesthetic_score_laion_v2 > 4.4]
df = df[df.clip_similarity_vitl14 > 0.245]
df = df[df.clip_similarity_vitb32 > 0.28]
df=df[["text"]]
df.to_parquet("full.parquet", index=False)
texts = df["text"].apply(lambda x: str(x))
texts = texts.to_list()

# df = pd.read_parquet("full.parquet")

clip = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").to(torch.float16)
tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

clip_forward = torch.compile(clip)

def get_embeds(batch):
    input_ids = tokenizer(batch, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(clip.device)
    with torch.no_grad():
        return clip_forward(input_ids).pooler_output

batch_size = 4096
cpu_steps = 64
all_embeds = []

embed_batch = []
for i in tqdm(range(0, len(df), batch_size)):
    batch = texts[i:i+batch_size]
    embeds = get_embeds(batch)
    embed_batch.append(embeds)
    if len(embed_batch) % cpu_steps == 0:
        embed_batch = torch.cat(embed_batch)
        embed_batch = embed_batch.to("cpu")
        all_embeds.append(embed_batch)
        embed_batch = []

if len(embed_batch) > 0:
    embed_batch = torch.cat(embed_batch)
    embed_batch = embed_batch.to("cpu")
    all_embeds.append(embed_batch)


torch.save(all_embeds, "all_embeds.pt")
