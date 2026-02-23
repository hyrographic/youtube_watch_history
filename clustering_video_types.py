import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path
import os
import shutil

from datetime import datetime

import re
from tqdm import tqdm

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

from scipy.ndimage import gaussian_filter1d

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

import umap
import numba

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

# ====== generate embeddings ======
# sent_transformer = SentenceTransformer("all-MiniLM-L6-v2") # fast model
sent_transformer = SentenceTransformer("all-mpnet-base-v2") # slow model

# - TITLE
_t_emb = sent_transformer.encode(titles.values, batch_size=256, show_progress_bar=True)
title_embeddings = pd.DataFrame(_t_emb, index=titles.index)
_cats = categories.loc[title_embeddings.index]
print('Title Embeddings: ', title_embeddings.shape)

palette = sns.color_palette('tab20', 16)
title_colours = {cat: palette[i] for i, cat in enumerate(_cats.unique())}
title_cmap = _cats.map(title_colours)

# - DESC
_desc_emb = sent_transformer.encode(desc.values, batch_size=256, show_progress_bar=True)
desc_embeddings = pd.DataFrame(_desc_emb, index=titles.index)
_cats = categories.loc[desc_embeddings.index]
print('Description Embeddings: ', desc_embeddings.shape)

# ====== combine embeddings ======
w_title = 2.0
w_desc  = 1.0

combined_embeddings = (title_embeddings * w_title + desc_embeddings * w_desc) / (w_title + w_desc)

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
mask = combined_embeddings.index.isin(single_month_ids)
masked_cats = categories.loc[mask]

# ====== UMAP boiler plate ======
n_components = 2
n_neighbors = 100
fit = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.0,
    n_components=n_components,
    metric='correlation'
)
umapped = fit.fit_transform(combined_embeddings);
umapped_colors = categories.map(title_colours)

umapped_month = umapped[mask]
month_colors = masked_cats.map(title_colours)

um = umapped
c = umapped_colors

fig = plt.figure(figsize=(19, 10), dpi=300)
if n_components == 2:
    ax = fig.add_subplot(111)
    ax.scatter(um[:,0], um[:,1], c=c, s=10, alpha=1)
if n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(um[:,0], um[:,1], um[:,2], c=c, s=5)

handles = [Patch(color=title_colours[cat], label=cat) for cat in title_colours]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=4, frameon=False)

# ====== UMAP Animation ======
def render_frames(df, umapped, combined_embeddings, categories, 
                  title_colours, output_dir, n_transition_frames=25):
    
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

    # Static background
    # ax.scatter(umapped[:,0], umapped[:,1], c=umapped_colors, s=10, alpha=0.05, linewidths=0)

    handles = [Patch(color=title_colours[cat], label=cat) for cat in title_colours]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False)

    def draw_frame(month_umapped, month_colors, alpha):
        # Remove previous highlights
        while len(ax.collections) > 2:  # keep background + legend
            ax.collections[-1].remove()

        if len(month_umapped) == 0:
            return

        # Glow layers
        for size, glow_alpha in [(200, 0.02), (100, 0.05), (50, 0.1), (20, 0.3)]:
            ax.scatter(month_umapped[:,0], month_umapped[:,1],
                      s=size, alpha=glow_alpha * alpha, 
                      color=month_colors, linewidths=0, zorder=3)
        # Core points
        ax.scatter(month_umapped[:,0], month_umapped[:,1],
                  s=15, alpha=alpha, color=month_colors, 
                  linewidths=0, zorder=4)

    def get_month_data(period):
        month_ids = df[df['date'].dt.to_period('W') == period]['id'].unique()
        mask = combined_embeddings.index.isin(month_ids)
        period_umapped = umapped[mask]
        period_colors = categories.loc[mask].map(title_colours).values
        print(f'period: {period}, umapped embedding values: {len(period_umapped)}')
        return period_umapped, period_colors

    for i, period in tqdm(enumerate(periods)):
        # t = pd.Period('2025-10-13/2025-10-19', 'W-SUN')
        period_umapped, period_colors = get_month_data(period)
        # Fade in
        for t in range(n_transition_frames):
            alpha = t / n_transition_frames
            draw_frame(period_umapped, period_colors, alpha)
            # display(fig)
            fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png', 
                       bbox_inches='tight', transparent=True)
            frame_idx += 1

        # Hold for 20 frames
        for _ in range(20):
            fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png',
                       bbox_inches='tight', transparent=True)
            frame_idx += 1

        # Fade out
        for t in range(n_transition_frames):
            alpha = 1 - (t / n_transition_frames)
            draw_frame(period_umapped, period_colors, alpha)
            fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.png',
                       bbox_inches='tight', transparent=True)
            frame_idx += 1

        print(f'Rendered {period} ({frame_idx} frames total)')

    plt.close(fig)
    print(f'Done — {frame_idx} frames saved to {output_dir}/')

render_frames(
    df=watch_data[watch_data['date'].dt.to_period('M')<'2023-03'],
    umapped=umapped, 
    combined_embeddings=combined_embeddings, 
    categories=categories, 
    title_colours=title_colours,
    output_dir='charts/frames_v3')