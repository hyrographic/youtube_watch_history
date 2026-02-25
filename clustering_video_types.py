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

categories = mdata_nlp['categories'].apply(lambda d: d[0] if isinstance(d, list) else 'None')
categories.name = 'category'

title_data = mdata_nlp[mdata_nlp['title'].notna()]
titles = title_data['title']

desc_data = mdata_nlp[mdata_nlp['description'].notna()]
desc = desc_data['description']

tags_data = mdata_nlp[(mdata_nlp['tags'].notna()) &(mdata_nlp['tags'].apply(len) > 0)]
tags = tags_data['tags'].str.join(' ')

# ====== generate embeddings ======
# sent_transformer = {'all-MiniLM-L6-v2':SentenceTransformer("all-MiniLM-L6-v2")} # fast model
# sent_transformer = SentenceTransformer("all-mpnet-base-v2") # slow model

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
m = 'all-mpnet-base-v2' #slow model
sent_transformer = create_transformer(m)

cat_embeddings = encode_cached(m, sent_transformer, categories.values, 'cat_emb.npy', categories.index)
title_embeddings = encode_cached(m, sent_transformer, titles.values,'title_emb.npy', titles.index)
desc_embeddings = encode_cached(m, sent_transformer, desc.values,'desc_emb.npy',  desc.index)
tags_embeddings = encode_cached(m, sent_transformer, tags.values,'tags_emb.npy',  tags.index)

print(f'cat={cat_embeddings.shape} title={title_embeddings.shape} desc={desc_embeddings.shape} tags={tags_embeddings.shape}')

_cats = categories.loc[title_embeddings.index]
palette = sns.color_palette('tab20', 16)
category_colours = {cat: palette[i] for i, cat in enumerate(categories.unique())}
category_cmap = categories.map(category_colours)
title_colours = {cat: palette[i] for i, cat in enumerate(_cats.unique())}
title_cmap = _cats.map(title_colours)

# ====== combine embeddings ======
# ---- embedding coverage ----
_embs = {'cat': cat_embeddings, 'title': title_embeddings, 'desc': desc_embeddings, 'tags': tags_embeddings}
_total = len(mdata_nlp)
for name, emb in _embs.items():
    print(f'{name:6s}: {len(emb):>5d} / {_total}')
print()
for (n1, e1), (n2, e2) in [
    (('cat',   cat_embeddings),   ('title', title_embeddings)),
    (('cat',   cat_embeddings),   ('desc',  desc_embeddings)),
    (('cat',   cat_embeddings),   ('tags',  tags_embeddings)),
    (('title', title_embeddings), ('desc',  desc_embeddings)),
    (('title', title_embeddings), ('tags',  tags_embeddings)),
    (('desc',  desc_embeddings),  ('tags',  tags_embeddings)),
]:
    n = len(e1.index.intersection(e2.index))
    print(f'{n1} ∩ {n2}: {n:>5d} / {_total}')
print()
_all4 = (cat_embeddings.index.intersection(title_embeddings.index)
                             .intersection(desc_embeddings.index)
                             .intersection(tags_embeddings.index))
_any1 = (cat_embeddings.index.union(title_embeddings.index)
                             .union(desc_embeddings.index)
                             .union(tags_embeddings.index))
print(f'all 4:  {len(_all4):>5d} / {_total}')
print(f'any 1:  {len(_any1):>5d} / {_total}')

# ! OPTION 1
# w_tags = 1.0
w_category = 0.25
w_title = 1.0
w_desc  = 0.25

# # Intersect indices so every row has all four embeddings present.
# # reindex + += propagates NaN: any video missing one feature silently becomes all-NaN.
common_idx = (cat_embeddings.index
              .intersection(title_embeddings.index)
              .intersection(desc_embeddings.index)
              .intersection(tags_embeddings.index))
print(f'Videos with all embeddings: {len(common_idx)} / {len(mdata_nlp)}')

# combined_embeddings = (
#     cat_embeddings.loc[common_idx]   * w_category +
#     title_embeddings.loc[common_idx] * w_title +
#     desc_embeddings.loc[common_idx]  * w_desc +
#     tags_embeddings.loc[common_idx]  * w_tags
# ) / (w_category + w_title + w_desc + w_tags)

# ! OPTION 2 — keep any video that has at least 1 embedding; average over available ones only
# Category is a discrete label (all "Music" videos share the exact same vector),
# so including it creates hard macro-cluster boundaries and suppresses sub-category structure.
# Use title + desc only for the embedding; keep categories only for colouring.
w_title = 1.0
w_desc  = 0.1
w_tags = 0.8

all_idx = (title_embeddings.index
           .union(title_embeddings.index)
           .union(desc_embeddings.index)
           .union(tags_embeddings.index)
           )

weighted_sum = pd.DataFrame(0.0, index=all_idx, columns=range(title_embeddings.shape[1]))
weight_total = pd.Series(0.0, index=all_idx)

for emb, w in [
    (title_embeddings, w_title),
    (desc_embeddings,  w_desc),
    (tags_embeddings,  w_tags)
]:
    present = all_idx.intersection(emb.index)
    weighted_sum.loc[present] += emb.loc[present].values * w
    weight_total.loc[present] += w

# Drop any row where no embedding was available (shouldn't happen given the union, but be safe)
valid = weight_total > 0
combined_embeddings = weighted_sum.loc[valid].div(weight_total.loc[valid], axis=0)
print(f'Videos with at least 1 embedding: {valid.sum()} / {len(mdata_nlp)}')

# ====== Embed on single combined string ======
_combined_idx = titles.index.union(tags_data['tags'].index)
_titles_aligned = titles.reindex(_combined_idx)
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

# ====== UMAP dimensionality reduction ======
# - TITLE
# reducer_2D_title = umap.UMAP(n_components=2, random_state=42, min_dist=0.3, spread=10.0)
# reduced_title_emb = reducer_2D_title.fit_transform(title_embeddings)
# reduced_title_emb = reducer_2D_title.transform(title_embeddings)

# plt.scatter(reduced_title_emb[:, 0],reduced_title_emb[:, 1], s=8, alpha=0.78, linewidths=0, color=c)
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP', fontsize=24);

# ====== Get single month sample ======
single_month = watch_data[watch_data['date'].dt.to_period('M') == '2024-04']
single_month_ids = single_month['id'].unique().tolist()
mask = composed_embeddings.index.isin(single_month_ids)
masked_cats = categories.loc[mask]

# ====== UMAP boiler plate ======
n_components = 2
n_neighbors = 8  # low = local structure / sub-clusters; high = global topology
fit = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.0,   # 0 = maximum internal compactness
    spread=3.0,     # spread clusters apart from each other (pairs with min_dist)
    n_components=n_components,
    metric='cosine'
)
umap_embeddings = fit.fit_transform(composed_embeddings);
umap_colors = categories.map(title_colours)

umap_month = umap_embeddings[mask]
month_colors = masked_cats.map(title_colours)

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

handles = [Patch(color=title_colours[cat], label=cat) for cat in title_colours]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, frameon=False)

# ====== clustering ======
clusterer = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
labels = clusterer.fit_predict(umap_embeddings)
clustered_embeddings = pd.DataFrame(umap_embeddings, index=composed_embeddings.index)
clustered_embeddings['hdbscan_label'] = labels
clustered_embeddings['channel'] = clustered_embeddings.index.map(watch_data.set_index('id')['channel_title'].to_dict()).fillna('None')
print('Number of HDBSCAN clusters: ', clustered_embeddings['hdbscan_label'].nunique())

# generate cluster colours
n_clusters = clustered_embeddings['channel'].nunique()
cmap = plt.cm.get_cmap('hsv', n_clusters)
colors = {label: cmap(i) for i, label in enumerate(sorted(set(labels)))}

# plot clusters
fig, ax = plt.subplots(figsize=(16, 10))
for label, group in clustered_embeddings.groupby('hdbscan_label'):
    ax.scatter(
        group[0], group[1],
        c=[colors[label]],
        s=3,
        alpha=0.5,
        linewidths=0,
        label=label if label != -1 else 'noise'
    )
# noise points (label == -1 from HDBSCAN) styled separately
noise = clustered_embeddings[clustered_embeddings['hdbscan_label'] == -1]
ax.scatter(noise[0], noise[1], c='lightgrey', s=2, alpha=0.3, linewidths=0)
plt.tight_layout()

# handles = [Patch(color=colors[cluster], label=cluster) for cluster in clustered_embeddings['hdbscan_label'].unique()]
# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, frameon=False)

plt.show()

# ====== inspect random cluster ======
def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in stopwords_list and len(w) > 1]

def sample_links(ids, n=2):
    sampled = random.sample(ids, min(n, len(ids)))
    return '  '.join(f'https://youtube.com/watch?v={i}' for i in sampled)
stopwords_list = set(stopwords.words('english'))

# ── Choose random cluster ────────────────────────────────────────────────────
valid_labels = [l for l in labels if l != -1]
s = random.sample(list(set(valid_labels)), 1)[0]
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
    for word in tokenize(title):
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
def render_frames(df, umapped, combined_embeddings, categories, title_colours, output_dir, n_fade_in_frames=20, n_fade_out_frames=30, window_size=3):
    """
    window_size: number of consecutive periods visible at once.
      For any frame showing 'current' period i, a point from period p has:
        offset = i - p
        offset = -1  → dim preview (appears 1 week before its recorded date)
        offset =  0  → full brightness (recorded date)
        offset 1..window_size-2 → linear fade toward 0
        offset >= window_size-1 → invisible

    n_fade_in_frames:  frames for a new period to rise from dim preview (offset -1) to full (offset 0)
    n_fade_out_frames: frames for existing periods to advance one offset step (age / dim)
    Total transition frames = max(n_fade_in_frames, n_fade_out_frames)
    """
    print('Number of df activities in umapped values: ', len(df[df['id'].isin(combined_embeddings.index)]))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    periods = sorted(df['date'].dt.to_period('W').unique())
    frame_idx = 0

    fig, ax = plt.subplots(figsize=(19, 10), dpi=150)
    fig.patch.set_alpha(1)
    ax.patch.set_alpha(0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Fix axis limits to the full UMAP extent so every frame has the same coordinate system.
    # Without this, matplotlib auto-scales per frame and the view shifts → misalignment.
    pad = 0.5
    ax.set_xlim(umapped[:, 0].min() - pad, umapped[:, 0].max() + pad)
    ax.set_ylim(umapped[:, 1].min() - pad, umapped[:, 1].max() + pad)

    # Static background
    # ax.scatter(umapped[:,0], umapped[:,1], c=umapped_colors, s=10, alpha=0.05, linewidths=0)

    handles = [Patch(color=title_colours[cat], label=cat) for cat in title_colours]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False)

    # Pre-cache per-period UMAP data so we don't re-query the DataFrame each frame
    period_data = []
    for period in periods:
        month_ids = df[df['date'].dt.to_period('W') == period]['id'].unique()
        mask = combined_embeddings.index.isin(month_ids)
        period_umapped = umapped[mask]
        period_colors = categories.loc[mask].map(title_colours).values
        period_data.append((period_umapped, period_colors))
        print(f'period: {period}, umapped embedding values: {len(period_umapped)}')

    def get_alpha_for_offset(offset):
        """
        offset = current_period_idx - point_period_idx
          -1  : dim preview (1 week before recorded date)
           0  : full brightness (recorded date)
          1..window_size-2 : linear fade from 1.0 toward 0
          >= window_size-1 : invisible
        """
        if offset == -1:
            return 0.25
        if offset == 0:
            return 1.0
        if 1 <= offset < window_size - 1:
            return 1.0 - offset / (window_size - 1)
        return 0.0

    def draw_rolling_frame(current_idx, progress_in=0.0, progress_out=0.0):
        """
        Draw the rolling window centred on current_idx.
          progress_in  [0,1]: how far the incoming period (offset -1 → 0) has brightened
          progress_out [0,1]: how far all existing periods (offset n → n+1) have aged/dimmed
        Both reach 1.0 when their respective transition is complete.
        """
        p_in  = (1 - np.cos(progress_in  * np.pi)) / 2
        p_out = (1 - np.cos(progress_out * np.pi)) / 2

        while ax.collections:
            ax.collections[-1].remove()

        # Periods that can be visible at current_idx or current_idx+1
        p_lo = max(0, current_idx - (window_size - 2))
        p_hi = min(len(periods), current_idx + 2)  # +1 preview, +1 for next-step preview

        for p_idx in range(p_lo, p_hi):
            offset = current_idx - p_idx
            # offset=-1 is the incoming preview — use progress_in
            # offset>=0 are existing/aging periods — use progress_out
            p = p_in if offset < 0 else p_out
            alpha = (1 - p) * get_alpha_for_offset(offset) \
                  + p       * get_alpha_for_offset(offset + 1)

            if alpha <= 0:
                continue

            p_umapped, p_colors = period_data[p_idx]
            if len(p_umapped) == 0:
                continue

            # Glow layers
            for size, glow_alpha in [(350, 0.05), (150, 0.05), (100, 0.1), (50, 0.3)]:
                ax.scatter(p_umapped[:, 0], p_umapped[:, 1],
                           s=size, alpha=glow_alpha * alpha,
                           color=p_colors, linewidths=0, zorder=3)
            # Core points
            ax.scatter(p_umapped[:, 0], p_umapped[:, 1],
                       s=50, alpha=alpha, color=p_colors,
                       linewidths=0, zorder=4)

    steps_in  = max(n_fade_in_frames  - 1, 1)
    steps_out = max(n_fade_out_frames - 1, 1)
    n_transition_frames = max(n_fade_in_frames, n_fade_out_frames)

    for i, period in tqdm(enumerate(periods)):
        # Transition from period i-1 → i.
        # Each progress independently clamps to 1.0 at its own rate.
        for t in range(n_transition_frames):
            if i == 0:
                draw_rolling_frame(0, progress_in=0.0, progress_out=0.0)
            else:
                p_in  = min(t / steps_in,  1.0)
                p_out = min(t / steps_out, 1.0)
                draw_rolling_frame(i - 1, progress_in=p_in, progress_out=p_out)
            fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png', transparent=True)
            frame_idx += 1

        # Hold on this period
        # for _ in range(9):
        #     draw_rolling_frame(i, progress_in=0.0, progress_out=0.0)
        #     fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png', transparent=True)
        #     frame_idx += 1

        print(f'Rendered {period} ({frame_idx} frames total)')

    plt.close(fig)
    print(f'Done — {frame_idx} frames saved to {output_dir}/')

render_frames(
    df=watch_data[watch_data['date'].dt.to_period('M')<'2024-01'],
    umapped=umapped,
    combined_embeddings=combined_embeddings,
    categories=categories,
    title_colours=title_colours,
    output_dir='charts/frames_v3',
    n_fade_in_frames=15,
    n_fade_out_frames=15,
)