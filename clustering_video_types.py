import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path
import os
import shutil
from collections import Counter
from datetime import datetime
import re
from tqdm import tqdm
import random
from importlib import reload

# data vis
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# machine learning
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap

# nlp
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# local imports
from src import read_and_load_data

# ====== read html data ======
data_paths = [
    'data/watch-history-a1.html',
    'data/watch-history-a2.html',
    'data/watch-history-b1.html'
]
# # read_and_load_data.read_html_files(paths=data_paths, save_json='y')

# ====== read JSON data ======
all_activities_df = read_and_load_data.read_parsed_data()
watch_data = read_and_load_data.filter_by_media(all_activities_df, 'watch')
read_and_load_data.filter_by_date(watch_data, 2023, 1, 11)

# ====== read metadata ======
metadata_paths = [
    'data/video_metadata.jsonl'
]
metadata_df = read_and_load_data.read_metadata(metadata_paths)
read_and_load_data.metadata_group_errors(metadata_df)

# ====== nlp prep ======
mdata_nlp = metadata_df.drop_duplicates(subset=['id']).dropna(subset=['id'])
mdata_nlp.set_index('id', inplace=True)

media_types_dict = mdata_nlp['media_type'].to_dict()

categories = mdata_nlp['categories'].apply(lambda d: d[0] if isinstance(d, list) else 'None')
categories.name = 'category'

title_data = mdata_nlp[mdata_nlp['title'].notna()]
titles = title_data['title']

desc_data = mdata_nlp[mdata_nlp['description'].notna()]
desc = desc_data['description']

tags_data = mdata_nlp[(mdata_nlp['tags'].notna()) &(mdata_nlp['tags'].apply(len) > 0)]
tags = tags_data['tags'].str.join(' ')

# ====== update stopwords ======
stopwords_list = list(set(stopwords.words('english')))

with open(Path('resources/english_stopwords.txt').resolve(), 'r') as f:
    kaggle_stopwords = f.read().split('\n')
stopwords_list.extend(kaggle_stopwords)

with open(Path('resources/custom_stopwords.txt').resolve(), 'r') as f:
    custom_stopwords = f.read().split('\n')
    add_to_stopwords = [x.strip().lower() for x in custom_stopwords if ':' not in x]
    str_replacements = {kv[0]:kv[1] for x in custom_stopwords if ':' in x for kv in [x.strip().lower().split(':')]}
stopwords_list.extend(add_to_stopwords)

# ====== clean tags data ======
tag_values = tags_data['tags'].explode()

tag_values = (
    tag_values
    .str.lower()
    .str.strip()
)

# remove stop words from tag values
tag_values = tag_values[~tag_values.isin(stopwords_list)]
# apply string replacements
tag_values = tag_values.map(str_replacements, na_action=None).fillna(tag_values)
# remove shorts
tag_values = tag_values.str.replace(' ?shorts ?', '', regex=1)

tag_values.dropna(inplace=True)

print('unique tags: ', tag_values.nunique())
tag_values.value_counts().iloc[50:].head(25)
# tag_values.value_counts(normalize=True).head(25)

tags_cleaned = tag_values.groupby(level=0).agg(list)

# ====== clean title data ======
def tokenise(text):
    if not isinstance(text, str):
        return []
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in stopwords_list and len(w) > 1]

title_values = title_data['title']
title_tokens = title_values.apply(tokenise).explode()

title_tokens = (
    title_tokens
    .str.lower()
    .str.strip()
)

# remove stop words from tag values
title_tokens = title_tokens[~title_tokens.isin(stopwords_list)]
# apply string replacements
title_tokens = title_tokens.map(str_replacements, na_action=None).fillna(title_tokens)
# remove shorts
title_tokens = title_tokens.str.replace(' ?shorts ?', '', regex=1)

title_tokens.dropna(inplace=True)

print('unique words: ', title_tokens.nunique())
title_tokens.value_counts().head(25)
# title_tokens.value_counts(normalize=True).head(25)

titles_cleaned = title_tokens.groupby(level=0).agg(lambda x: ' '.join(x))

# ====== generate embeddings ======
def create_transformer(model):
    return SentenceTransformer(model)

def encode_cached(m, transformer, values, file_name, index):
    fp = f'data/embeddings_cache/{m}/{file_name}'
    os.makedirs('/'.join(fp.split('/')[:-1]), exist_ok=True)
    if os.path.exists(fp):
        print(f'Loading cached {file_name}')
        return pd.DataFrame(np.load(fp), index=index)
    print(f'Encoding {file_name}...')
    emb = transformer.encode(values, batch_size=256, show_progress_bar=True)
    np.save(fp, emb)
    return pd.DataFrame(emb, index=index)

# m = 'all-MiniLM-L6-v2' # fast model
# m = 'all-mpnet-base-v2' #slow model
m = 'distilbert-base-nli-mean-tokens'
sent_transformer = create_transformer(m)

# ====== Embed on single combined string ======
_combined_idx = titles_cleaned.index.union(tags_cleaned.index)
_titles_aligned = titles_cleaned.reindex(_combined_idx)
_tags_aligned   = tags_data['tags'].reindex(_combined_idx)

def compose(vid_id):
    parts = []
    tag_list = _tags_aligned.loc[vid_id]
    if isinstance(tag_list, list) and tag_list:
        parts.append(' '.join(tag_list[:10]))
    title = _titles_aligned.loc[vid_id]
    if isinstance(title, str) and title:
        parts.append(title)
    return ' | '.join(parts)

composed = pd.Series([compose(i) for i in _combined_idx], index=_combined_idx)
composed_embeddings = encode_cached(m, sent_transformer, composed.tolist(), 'composed_emb.npy', composed.index)
composed_cats = categories.loc[composed.index].copy()

# ====== UMAP dimensionality reduction ======
# ====== Get single month sample ======
single_month = watch_data[watch_data['date'].dt.to_period('M') == '2024-04']
single_month_ids = single_month['id'].unique().tolist()
mask = composed_embeddings.index.isin(single_month_ids)
masked_cats = categories.loc[mask]

# ====== UMAP boiler plate ======
# video_embeddings = composed_embeddings[composed_embeddings.index.map(media_types_dict) == 'video']
n_components = 2
n_neighbors = 16  # low = local structure / sub-clusters; high = global topology
fit = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.0,   # 0 = maximum internal compactness
    spread=3.0,     # spread clusters apart from each other (pairs with min_dist)
    n_components=n_components,
    metric='correlation'
)
umap_embeddings = fit.fit_transform(composed_embeddings);

palette = sns.color_palette('tab20', 16)
colours = {cat: palette[i] for i, cat in enumerate(composed_cats.unique())}
umap_colors = composed_cats.map(colours)

# umap_month = umap_embeddings[mask]
# month_colors = masked_cats.map(colours)

um = umap_embeddings
c = umap_colors

fig = plt.figure(figsize=(19, 10), dpi=300)
if n_components == 2:
    ax = fig.add_subplot(111)
    ax.scatter(um[:,0], um[:,1], c=c, s=3, alpha=0.5)
    # ax.set_ylim(-10, 10)
    # ax.set_xlim(0, 20)
if n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(um[:,0], um[:,1], um[:,2], c=c, s=5)

handles = [Patch(color=colours[cat], label=cat) for cat in colours]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, frameon=False)

# ====== HDBSCAN clustering ======
clusterer = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
labels = clusterer.fit_predict(umap_embeddings)
clustered_embeddings = pd.DataFrame(umap_embeddings, index=composed_embeddings.index)
clustered_embeddings['hdbscan_label'] = labels
# clustered_embeddings['channel'] = clustered_embeddings.index.map(watch_data.set_index('id')['channel_title'].to_dict()).fillna('None')
# clustered_embeddings['categories'] = clustered_embeddings.index.map(categories).fillna('None')
clustered_embeddings['duration'] = clustered_embeddings.index.map(mdata_nlp['duration']).fillna('None')
print('Number of HDBSCAN clusters: ', clustered_embeddings['hdbscan_label'].nunique())
print('Cluster Sizes: ', clustered_embeddings['hdbscan_label'].value_counts().head(10))

# generate cluster colours
unique_labels = sorted(l for l in set(clustered_embeddings['hdbscan_label']) if l != -1)
n_clusters = len(unique_labels)
cmap = plt.cm.get_cmap('hsv', n_clusters)
hdbscan_colours = {label: cmap(i) for i, label in enumerate(unique_labels)}

# plot clusters
fig, ax = plt.subplots(figsize=(16, 10))
for label, group in clustered_embeddings[clustered_embeddings['hdbscan_label'] != -1].groupby('hdbscan_label'):
    ax.scatter(
        group[0], group[1],
        color=hdbscan_colours[label],
        s=3,
        alpha=0.5,
        linewidths=0,
    )
# noise points (label == -1 from HDBSCAN) styled separately
noise = clustered_embeddings[clustered_embeddings['hdbscan_label'] == -1]
ax.scatter(noise[0], noise[1], c='lightgrey', s=2, alpha=0.3, linewidths=0)

plt.yticks(np.arange(-20, 35, 2.5))
plt.xticks(np.arange(-20, 40, 2.5))
plt.grid(True)
plt.tight_layout()

# handles = [Patch(color=hdbscan_colours[cluster], label=cluster) for cluster in clustered_embeddings['hdbscan_label'].unique()]
# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, frameon=False)

plt.show()

# ====== inspect random cluster ======
def sample_links(ids, n=2):
    sampled = random.sample(ids, min(n, len(ids)))
    return '  '.join(f'https://youtube.com/watch?v={i}' for i in sampled)

# ── Choose random cluster ────────────────────────────────────────────────────
valid_labels = [l for l in labels if l != -1]
s = random.sample(list(set(valid_labels)), 1)[0]
s = 22
s_df = clustered_embeddings[clustered_embeddings['hdbscan_label'] == s].copy()
s_ids = s_df.index.tolist()
print(f'Videos in cluster {s}: ', len(s_df))
s_watch_data = watch_data[watch_data['id'].isin(s_ids)].copy()
print('Watched from: ', s_watch_data['date'].min(), ' to ', s_watch_data['date'].max())

# ── Weekly watch bar chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 3))

weekly = s_watch_data.set_index('date').resample('W')['id'].count()
ax.bar(weekly.index, weekly.values, width=6, color='steelblue', alpha=0.8)

ax.set_xlabel('Week')
ax.set_ylabel('Videos watched')
ax.set_title(f'Cluster {s} — weekly watch activity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ── Cluster highlight scatter ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))

mask = clustered_embeddings['hdbscan_label'] != s
ax.scatter(
    clustered_embeddings.loc[mask, 0],
    clustered_embeddings.loc[mask, 1],
    c='grey', s=2, alpha=0.3, linewidths=0
)

ax.scatter(
    s_df[0], s_df[1],
    c='crimson', s=8, alpha=0.9, linewidths=0,
    label=f'Cluster {s} (n={len(s_df)})'
)

ax.legend(loc='upper right')
ax.set_title(f'Cluster {s} — position in embedding space')
ax.axis('off')
plt.tight_layout()
plt.show()

# ── Channels ────────────────────────────────────────────────────────────────
print('\n── Top Channels ──')
channel_counts = s_watch_data['channel_title'].value_counts().head(10)
for channel, count in channel_counts.items():
    print(f'  {count:>4}  {channel}')

# ── Title word frequencies ───────────────────────────────────────────────────
cluster_meta = mdata_nlp[mdata_nlp.index.isin(s_ids)]

word_to_ids = {}
for vid_id, title in cluster_meta['title'].dropna().items():
    for word in tokenise(title):
        word_to_ids.setdefault(word, []).append(vid_id)

title_words = Counter({w: len(ids) for w, ids in word_to_ids.items()})

print('\n── Top Title Words ──')
for word, count in title_words.most_common(20):
    print(f'  {count:>4}  {word:<25} {sample_links(word_to_ids[word])}')

# ── Tag frequencies ──────────────────────────────────────────────────────────
tag_to_ids = {}
for vid_id, tags in cluster_meta['tags'].dropna().items():
    if isinstance(tags, list):
        tag_list = [t.lower().strip() for t in tags]
    elif isinstance(tags, str):
        tag_list = [t.lower().strip() for t in tags.split(',')]
    else:
        continue
    for tag in tag_list:
        tag_to_ids.setdefault(tag, []).append(vid_id)

tag_counts = Counter({t: len(ids) for t, ids in tag_to_ids.items()})

print('\n── Top Tags ──')
for tag, count in tag_counts.most_common(20):
    print(f'  {count:>4}  {tag:<25} {sample_links(tag_to_ids[tag])}')


# ====== Clustering Animation ======
import animation_rendering
reload(animation_rendering)

renderer = animation_rendering.UMAPAnimationRenderer(
    fps=30,
    seconds_per_day=0.2,
    window_size=15,
    glow_size=1.5,
    scale_by_duration=True,
    noise_duration_min_size=4,
    noise_duration_max_size=150
)

render_range = watch_data[(watch_data['date'].dt.to_period('M')>='2024-01') & (watch_data['date'].dt.to_period('M')<='2024-12')].copy()

render_range = watch_data[(watch_data['date'].dt.to_period('D')>='2025-01-01') & (watch_data['date'].dt.to_period('D')<='2025-03-01')].copy()

# Quick single-frame preview — run this to iterate on visuals without a full render
# renderer.sample_frame(render_range, clustered_embeddings, hdbscan_colours, day_idx=0, save_path='charts/sample_frame.svg')

renderer.render(
    df=render_range,
    combined_embeddings=clustered_embeddings,
    hdbscan_colours=hdbscan_colours,
    output_dir='charts/frames_v9'
)