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
import numba

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

from sklearn.decomposition import PCA
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
    return SentenceTransformer(model, trust_remote_code=True)

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
# m = 'distilbert-base-nli-mean-tokens'
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

m = 'nomic-ai/nomic-embed-text-v1.5'
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

composed = pd.Series([compose(i) for i in _combined_idx], index=_combined_idx).sample(round(len(_combined_idx)*0.3)) # ! TESTING SAMPLE

composed_embeddings = encode_cached(m, sent_transformer, composed.tolist(), 'nomic_1.npy', composed.index)
composed_cats = categories.loc[composed.index].copy()

#%%
# ====== UMAP dimensionality reduction ======
# ====== Get single month sample ======
single_month = watch_data[watch_data['date'].dt.to_period('M') == '2024-04']
single_month_ids = single_month['id'].unique().tolist()
mask = composed_embeddings.index.isin(single_month_ids)
masked_cats = composed_cats.loc[mask]

# video_embeddings = composed_embeddings[composed_embeddings.index.map(media_types_dict) == 'video']
@numba.njit()
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

@numba.njit()
def lightness(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    return (cmax + cmin) / 2.0

@numba.njit()
def saturation(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    chroma = cmax - cmin
    light = lightness(r, g, b)
    if light == 1:
        return 0
    else:
        return chroma / (1 - abs(2*light - 1))

@numba.njit()
def hsl_dist(a, b):
    a_sat = saturation(a[0], a[1], a[2])
    b_sat = saturation(b[0], b[1], b[2])
    a_light = lightness(a[0], a[1], a[2])
    b_light = lightness(b[0], b[1], b[2])
    a_hue = hue(a[0], a[1], a[2])
    b_hue = hue(b[0], b[1], b[2])
    return (a_sat - b_sat)**2 + (a_light - b_light)**2 + (((a_hue - b_hue) % 6) / 6.0)


n_components = 2
n_neighbors = 16  # low = local structure / sub-clusters; high = global topology

dim_reduction_model = umap.UMAP(output_metric='haversine', random_state=42).fit(composed_embeddings)

# dim_reduction_model = PCA(n_components==n_components)

# dim_reduction_model = umap.UMAP(
#     n_neighbors=n_neighbors,
#     min_dist=0.0,   # 0 = maximum internal compactness
#     spread=3.0,     # spread clusters apart from each other (pairs with min_dist)
#     n_components=n_components,
#     # metric='correlation'
# )
umap_embeddings = pd.DataFrame(dim_reduction_model.fit_transform(composed_embeddings), index=composed_embeddings.index)

palette = sns.color_palette('tab20', 16)
colours = {cat: palette[i] for i, cat in enumerate(composed_cats.unique())}
umap_colors = composed_cats.map(colours)

# umap_month = umap_embeddings[mask]
# month_colors = masked_cats.map(colours)

um = umap_embeddings.values
c = umap_colors

# exclude = composed_cats[~composed_cats.isin([
#     'People & Blogs'
# ])]
# um = umap_embeddings.loc[exclude.index].values
# c = composed_cats.loc[exclude.index].map(colours)

x = np.sin(dim_reduction_model.embedding_[:, 0]) * np.cos(dim_reduction_model.embedding_[:, 1])
y = np.sin(dim_reduction_model.embedding_[:, 0]) * np.sin(dim_reduction_model.embedding_[:, 1])
z = np.cos(dim_reduction_model.embedding_[:, 0])

# fig = plt.figure(figsize=(19, 10), dpi=300)
# if n_components == 2:
#     ax = fig.add_subplot(111)
#     ax.scatter(um[:,0], um[:,1], c=c, s=3, alpha=0.5)
#     # ax.set_ylim(-10, 10)
#     # ax.set_xlim(0, 20)

fig = plt.figure(figsize=(19, 10), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=c, s=1, alpha=0.25, cmap='Spectral')

handles = [Patch(color=colours[cat], label=cat) for cat in colours]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, frameon=False)

##

#%%
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


#%%
# ====== inspect random cluster ======
def sample_links(ids, n=2):
    sampled = random.sample(ids, min(n, len(ids)))
    return '  '.join(f'https://youtube.com/watch?v={i}' for i in sampled)

# ── Choose random cluster ────────────────────────────────────────────────────
valid_labels = [l for l in labels if l != -1]
s = random.sample(list(set(valid_labels)), 1)[0]
# s = 49
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


# ====== Region & cluster inspection helpers ======

def find_clusters_in_region( x_bounds, y_bounds, day=None, window_days=15):
    """
    Find clusters whose points fall within a spatial region of the UMAP embedding.

    Parameters
    ----------
    x_bounds : (float, float)   x_min, x_max of the region of interest.
    y_bounds : (float, float)   y_min, y_max of the region of interest.
    day      : str or Period, optional
        If given (e.g. '2024-03-15'), restrict to video IDs that were watched
        within window_days days ending on this date.
    window_days : int
        Width of the watch-activity window when day is specified.

    Returns
    -------
    list of cluster labels (ints) found in the region, sorted by point count.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    subset = clustered_embeddings
    if day is not None:
        end   = pd.Period(day, 'D')
        start = end - window_days
        active_ids = set(
            watch_data[watch_data['date'].dt.to_period('D').between(str(start), str(end))]['id']
        )
        subset = subset[subset.index.isin(active_ids)]

    spatial_mask = (
        (subset[0] >= x_min) & (subset[0] <= x_max) &
        (subset[1] >= y_min) & (subset[1] <= y_max)
    )
    region = subset[spatial_mask]

    cluster_counts = (
        region[region['hdbscan_label'] != -1]['hdbscan_label']
        .value_counts()
    )
    found_labels = cluster_counts.index.tolist()

    # ── Zoomed scatter ────────────────────────────────────────────────────────
    BG = '#0f0f0f'
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={'width_ratios': [1, 2]},
                             facecolor=BG)

    # Left: full overview with region box — no decorations
    ax_full = axes[0]
    ax_full.set_facecolor(BG)
    ax_full.scatter(clustered_embeddings[0], clustered_embeddings[1],
                    c='#3a3a3a', s=1, alpha=0.6, linewidths=0)
    for label in found_labels:
        grp = clustered_embeddings[clustered_embeddings['hdbscan_label'] == label]
        ax_full.scatter(grp[0], grp[1], color=hdbscan_colours.get(label, (0.5, 0.5, 0.5)),
                        s=3, alpha=0.7, linewidths=0)
    rect = matplotlib.patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=1.5, edgecolor='white', facecolor='none')
    ax_full.add_patch(rect)
    ax_full.axis('off')
    title = f'day={day}  ±{window_days}d' if day else 'all time'
    ax_full.set_title(f'Overview — {title}', fontsize=9, color='white')

    # Right: zoomed region with grid, axis labels, centroid badges
    ax_zoom = axes[1]
    ax_zoom.set_facecolor(BG)
    x_pad = (x_max - x_min) * 0.25
    y_pad = (y_max - y_min) * 0.25
    ax_zoom.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_zoom.grid(True, color='white', alpha=0.1, linewidth=0.6, zorder=0)
    ax_zoom.set_xlabel('UMAP 1', color='white', fontsize=9)
    ax_zoom.set_ylabel('UMAP 2', color='white', fontsize=9)
    ax_zoom.tick_params(colors='white', labelsize=7)
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('#444444')

    noise = region[region['hdbscan_label'] == -1]
    if len(noise):
        ax_zoom.scatter(noise[0], noise[1], c='#555555', s=4, alpha=0.4, linewidths=0, zorder=1)
    for label in found_labels:
        grp = region[region['hdbscan_label'] == label]
        full_grp_ids = clustered_embeddings[clustered_embeddings['hdbscan_label'] == label].index
        color = hdbscan_colours.get(label, (0.5, 0.5, 0.5))
        ax_zoom.scatter(grp[0], grp[1], color=color, s=14, alpha=0.85, linewidths=0, zorder=2)
        cx, cy = float(grp[0].mean()), float(grp[1].mean())
        # Top channel for this cluster
        ch_counts = watch_data[watch_data['id'].isin(full_grp_ids)]['channel_title'].value_counts()
        top_ch = ch_counts.index[0][:20] if len(ch_counts) else ''
        ann_text = f'#{label}  n={len(full_grp_ids)}\n{top_ch}'
        ax_zoom.annotate(ann_text, (cx, cy), fontsize=8, color='white',
                         fontweight='bold', ha='center', va='center', zorder=3,
                         linespacing=1.4,
                         bbox=dict(boxstyle='round,pad=0.35', fc=color, ec='none', alpha=0.85))
    ax_zoom.set_title(f'Region  x={x_bounds}  y={y_bounds}', fontsize=9, color='white')

    plt.tight_layout()
    plt.show()

    print(f'\nClusters in region ({len(region)} total points):')
    for label, count in cluster_counts.items():
        print(f'  Cluster {label:>3}: {count} points')
    if len(noise):
        print(f'  noise     : {len(noise)} points')

    return found_labels


def inspect_cluster(s, n_links=2):
    """
    Print and plot summary information for a single HDBSCAN cluster.

    Parameters
    ----------
    s       : int   Cluster label (as assigned by HDBSCAN).
    n_links : int   Number of sample YouTube links to show per word/tag row.
    """
    def _links(ids):
        sampled = random.sample(ids, min(n_links, len(ids)))
        return '  '.join(f'https://youtube.com/watch?v={i}' for i in sampled)

    s_df = clustered_embeddings[clustered_embeddings['hdbscan_label'] == s].copy()
    s_ids = s_df.index.tolist()
    s_watch = watch_data[watch_data['id'].isin(s_ids)].copy()

    print(f'Cluster {s}  —  {len(s_df)} videos')
    print(f'Watched from {s_watch["date"].min():%Y-%m-%d} to {s_watch["date"].max():%Y-%m-%d}')

    # ── Weekly watch bar chart — stacked by cluster ───────────────────────────
    id_to_label = clustered_embeddings['hdbscan_label'].to_dict()
    all_watch = watch_data.copy()
    all_watch['cluster'] = all_watch['id'].map(id_to_label).fillna(-1).astype(int)

    # Restrict to the cluster's active date range for readability
    ranged = all_watch[
        (all_watch['date'] >= s_watch['date'].min()) &
        (all_watch['date'] <= s_watch['date'].max())
    ]

    weekly_by_cluster = (
        ranged.set_index('date')
        .groupby([pd.Grouper(freq='W'), 'cluster'])['id']
        .count()
        .unstack(fill_value=0)
    )
    weeks = weekly_by_cluster.index

    # Cluster s at the bottom so its bar height is directly readable
    # Top 7 other clusters above it, everything else lumped as grey on top
    other_cols = [c for c in weekly_by_cluster.columns if c != s and c != -1]
    top_others = weekly_by_cluster[other_cols].sum().nlargest(7).index.tolist()
    lumped_cols = [c for c in weekly_by_cluster.columns if c not in top_others and c != s]

    fig, ax = plt.subplots(figsize=(18, 4))
    bottom = np.zeros(len(weeks))

    color_s = hdbscan_colours.get(s, (0.5, 0.5, 0.9))
    s_vals = weekly_by_cluster[s].values if s in weekly_by_cluster.columns else np.zeros(len(weeks))
    ax.bar(weeks, s_vals, bottom=bottom, width=6, color=color_s, alpha=0.95, label=f'Cluster {s}')
    bottom += s_vals

    for label in top_others:
        color = hdbscan_colours.get(label, (0.5, 0.5, 0.5))
        vals = weekly_by_cluster[label].values
        ax.bar(weeks, vals, bottom=bottom, width=6, color=color, alpha=0.5)
        bottom += vals

    if lumped_cols:
        lumped_vals = weekly_by_cluster[lumped_cols].sum(axis=1).values
        ax.bar(weeks, lumped_vals, bottom=bottom, width=6, color='#555555', alpha=0.35, label='other')
        bottom += lumped_vals

    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.set_xlabel('Week')
    ax.set_ylabel('Videos watched')
    ax.set_title(f'Cluster {s} — weekly watch activity vs other clusters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ── Closest clusters ─────────────────────────────────────────────────────
    all_labels = [l for l in clustered_embeddings['hdbscan_label'].unique() if l != -1 and l != s]
    centroids = {
        l: (clustered_embeddings.loc[clustered_embeddings['hdbscan_label'] == l, 0].mean(),
            clustered_embeddings.loc[clustered_embeddings['hdbscan_label'] == l, 1].mean())
        for l in all_labels
    }
    s_cx, s_cy = s_df[0].mean(), s_df[1].mean()
    dists = sorted(
        ((np.hypot(cx - s_cx, cy - s_cy), l) for l, (cx, cy) in centroids.items())
    )
    n_close = 5
    closest = dists[:n_close]

    print('\n── Closest Clusters ──')
    for dist, l in closest:
        n_l = (clustered_embeddings['hdbscan_label'] == l).sum()
        print(f'  #{l:>4}  dist={dist:.3f}  n={n_l}')

    # ── Cluster highlight scatter ─────────────────────────────────────────────
    close_labels = [l for _, l in closest]
    fig, ax = plt.subplots(figsize=(18, 6))

    # Grey background (all other points)
    rest_mask = ~clustered_embeddings['hdbscan_label'].isin([s] + close_labels)
    ax.scatter(clustered_embeddings.loc[rest_mask, 0], clustered_embeddings.loc[rest_mask, 1],
               c='grey', s=2, alpha=0.2, linewidths=0)

    # Closest clusters in their own colours
    for dist, l in closest:
        l_df = clustered_embeddings[clustered_embeddings['hdbscan_label'] == l]
        color = hdbscan_colours.get(l, (0.6, 0.6, 0.6))
        ax.scatter(l_df[0], l_df[1], c=[color], s=5, alpha=0.6, linewidths=0)
        cx, cy = centroids[l]
        ax.annotate(f'#{l}', (cx, cy), color='white', fontsize=7,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.85, lw=0))

    # Target cluster on top
    color_s = hdbscan_colours.get(s, (0.86, 0.08, 0.24))
    ax.scatter(s_df[0], s_df[1], c=[color_s], s=10, alpha=0.95, linewidths=0,
               label=f'Cluster {s} (n={len(s_df)})')
    ax.annotate(f'#{s}', (s_cx, s_cy), color='white', fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc=color_s, alpha=0.95, lw=0))

    ax.legend(loc='upper right', frameon=False, labelcolor='white',
              facecolor='#0f0f0f', fontsize=8)
    ax.set_facecolor('#0f0f0f')
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_title(f'Cluster {s} — position in embedding space', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    # ── Top channels ─────────────────────────────────────────────────────────
    print('\n── Top Channels ──')
    for channel, count in s_watch['channel_title'].value_counts().head(10).items():
        print(f'  {count:>4}  {channel}')

    # ── Title word frequencies ────────────────────────────────────────────────
    cluster_meta = mdata_nlp[mdata_nlp.index.isin(s_ids)]
    word_to_ids = {}
    for vid_id, title in cluster_meta['title'].dropna().items():
        for word in tokenise(title):
            word_to_ids.setdefault(word, []).append(vid_id)
    title_words = Counter({w: len(ids) for w, ids in word_to_ids.items()})

    print('\n── Top Title Words ──')
    for word, count in title_words.most_common(20):
        print(f'  {count:>4}  {word:<25} {_links(word_to_ids[word])}')

    # ── Tag frequencies ───────────────────────────────────────────────────────
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
        print(f'  {count:>4}  {tag:<25} {_links(tag_to_ids[tag])}')
#%%

# ====== Clustering Animation ======
import animation_rendering
reload(animation_rendering)

renderer = animation_rendering.UMAPAnimationRenderer(
    fps=30,
    seconds_per_day=0.2,
    window_size=15,
    glow_size=1.5,
    scale_by_duration=True,
    noise_duration_min_size=0.5,
    noise_duration_max_size=80
)

render_range = watch_data[(watch_data['date'].dt.to_period('M')>='2024-01') & (watch_data['date'].dt.to_period('M')<='2024-12')].copy()

render_range = watch_data[(watch_data['date'].dt.to_period('D')>='2025-01-01') & (watch_data['date'].dt.to_period('D')<='2025-03-01')].copy()

# Quick single-frame preview — run this to iterate on visuals without a full render
renderer.sample_frame(render_range, clustered_embeddings, hdbscan_colours, day_idx=72, save_path='charts/sample_frame.svg')

renderer.frame_to_date(72)


renderer.render(
    df=render_range,
    combined_embeddings=clustered_embeddings,
    hdbscan_colours=hdbscan_colours,
    output_dir='charts/frames_v9'
)