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
clustered_embeddings['channel'] = clustered_embeddings.index.map(watch_data.set_index('id')['channel_title'].to_dict()).fillna('None')
clustered_embeddings['categories'] = clustered_embeddings.index.map(categories).fillna('None')
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
s = 195
s_df = clustered_embeddings[clustered_embeddings['hdbscan_label'] == s].copy()
s_ids = s_df.index.tolist()
print('Videos in cluster: ', len(s_df))
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


# ====== UMAP Animation ======
def render_frames(df, umapped, combined_embeddings, output_dir,
                  fps=30, seconds_per_day=0.2,
                  window_size=7, noise_window_multiplier=4,
                  glow_size=1.0):
    """
    fps / seconds_per_day → frames_per_day = round(fps * seconds_per_day)
      e.g. 30fps × 0.2s = 6 frames per day.

    Within each day, watches are revealed in chronological order across
    frames_per_day frames. The rolling window (window_size days) fades
    previous days by day-offset, not frame-offset.

    Stitch output with:
      ffmpeg -r {fps} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4
    """
    frames_per_day = max(1, round(fps * seconds_per_day))
    noise_window   = window_size * noise_window_multiplier
    print(f'frames_per_day={frames_per_day}  window_size={window_size}  noise_window={noise_window}')
    print(f'Activities with embeddings: {len(df[df["id"].isin(combined_embeddings.index)])}')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    days = sorted(df['date'].dt.to_period('D').unique())

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)  # 3840×2160 (4K)
    fig.patch.set_alpha(1)
    ax.patch.set_alpha(0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    pad = 0.5
    x_min, x_max = umapped[0].min() - pad, umapped[0].max() + pad
    y_min, y_max = umapped[1].min() - pad, umapped[1].max() + pad
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_h, y_h = (x_max - x_min) / 2 / 1.35, (y_max - y_min) / 2 / 1.35
    ax.set_xlim(x_c - x_h, x_c + x_h)
    ax.set_ylim(y_c - y_h, y_c + y_h)
    ax.set_autoscale_on(False)  # prevent scatter calls inside draw_frame from resetting limits

    # ── Pre-cache: one entry per day, watches in chronological order ──────────
    day_ordered_ids = []  # list[list[str]]
    day_umap_data   = []  # list[(umap_df, colors_arr, is_noise_arr)]

    for day in tqdm(days, desc='Pre-caching days'):
        day_df = df[df['date'].dt.to_period('D') == day].sort_values('date')
        seen, ordered = set(), []
        for vid_id in day_df['id']:
            if vid_id not in seen and vid_id in combined_embeddings.index:
                seen.add(vid_id)
                ordered.append(vid_id)
        day_ordered_ids.append(ordered)

        if ordered:
            umap_sub = combined_embeddings.loc[ordered]
            labels   = umap_sub['hdbscan_label'].values
            colors   = np.array([hdbscan_colours.get(l, (0.6, 0.6, 0.6, 1.0)) for l in labels])
            is_noise = (labels == -1)
        else:
            umap_sub = combined_embeddings.iloc[:0]  # empty
            colors   = np.empty((0, 4))
            is_noise = np.empty(0, dtype=bool)
        day_umap_data.append((umap_sub, colors, is_noise))
        print(f'  {day}: {len(ordered)} embeddable watches')

    # ── Alpha functions (offset = day_idx_current - day_idx_past) ────────────
    def get_alpha_for_offset(offset):
        # power > 1 → fast initial decay, then nearly flat for last ~25-30%
        if offset == 0:
            return 1.0
        if 1 <= offset < window_size:
            return (1.0 - offset / window_size) ** 2.0
        return 0.0

    def get_noise_alpha_for_offset(offset):
        if offset == 0:
            return 0.10
        if 1 <= offset < noise_window:
            return 0.10 * (1.0 - offset / noise_window) ** 0.4
        return 0.0

    # Glow layers: many thin concentric rings with exponential alpha falloff
    # gives a soft gaussian-like bloom rather than hard rings.
    # sizes decrease toward core; alphas increase (outer is barely visible)
    _glow_layers = list(zip(
        [s * glow_size for s in [130, 105, 83, 64, 49, 37, 27, 19, 13]],
        [0.004, 0.007, 0.011, 0.018, 0.03, 0.05, 0.08, 0.13, 0.20],
    ))

    # ── Draw a single frame ───────────────────────────────────────────────────
    def draw_frame(day_idx, n_visible):
        """
        day_idx   : which day is 'current'
        n_visible : how many of today's time-ordered watches to show (0 → N)
        """
        while ax.collections:
            ax.collections[-1].remove()

        # Noise pass — wide window, no glow, dim
        for d_idx in range(max(0, day_idx - noise_window + 1), day_idx + 1):
            offset = day_idx - d_idx
            alpha  = get_noise_alpha_for_offset(offset)
            if alpha <= 0:
                continue
            umap_sub, colors, is_noise = day_umap_data[d_idx]
            n = n_visible if d_idx == day_idx else len(umap_sub)
            if n == 0 or not is_noise[:n].any():
                continue
            ax.scatter(umap_sub[0].values[:n][is_noise[:n]],
                       umap_sub[1].values[:n][is_noise[:n]],
                       s=12, alpha=alpha, color='grey', linewidths=0, zorder=1)

        # Cluster pass — normal window, glow, full brightness
        for d_idx in range(max(0, day_idx - window_size + 1), day_idx + 1):
            offset = day_idx - d_idx
            alpha  = get_alpha_for_offset(offset)
            if alpha <= 0:
                continue
            umap_sub, colors, is_noise = day_umap_data[d_idx]
            n         = n_visible if d_idx == day_idx else len(umap_sub)
            non_noise = ~is_noise[:n]
            if n == 0 or not non_noise.any():
                continue
            xu = umap_sub[0].values[:n][non_noise]
            yu = umap_sub[1].values[:n][non_noise]
            pc = colors[:n][non_noise]
            for size, ga in _glow_layers:
                ax.scatter(xu, yu, s=size, alpha=ga * alpha, color=pc, linewidths=0, zorder=3)
            ax.scatter(xu, yu, s=20 * glow_size, alpha=alpha, color=pc, linewidths=0, zorder=4)

        # Re-enforce limits — savefig can re-expand them even with autoscale off
        ax.set_xlim(x_c - x_h, x_c + x_h)
        ax.set_ylim(y_c - y_h, y_c + y_h)

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    for day_idx, day in tqdm(enumerate(days), total=len(days), desc='Rendering'):
        n_today = len(day_ordered_ids[day_idx])
        for f in range(frames_per_day):
            n_visible = round((f + 1) / frames_per_day * n_today)
            draw_frame(day_idx, n_visible)
            fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png', transparent=True)
            frame_idx += 1
        print(f'Rendered {day} ({frame_idx} frames total)')

    plt.close(fig)
    print(f'Done — {frame_idx} frames at {fps}fps → {frame_idx/fps:.1f}s saved to {output_dir}/')

render_range = watch_data[(watch_data['date'].dt.to_period('M')>='2024-01') & (watch_data['date'].dt.to_period('M')<='2024-12')].copy()

render_range = watch_data[(watch_data['date'].dt.to_period('D')>='2024-01-01') & (watch_data['date'].dt.to_period('D')<='2024-02-01')].copy()

render_frames(
    df=watch_data,
    umapped=clustered_embeddings,
    combined_embeddings=clustered_embeddings,
    output_dir='charts/frames_v8',
    fps=30,
    seconds_per_day=0.2,
    window_size=15,
    glow_size=1.5
)