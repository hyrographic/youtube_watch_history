import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path

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

from scipy.ndimage import gaussian_filter1d

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import umap

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
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
# all-mpnet-base-v2

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

# ====== dimensionality reduction ======
# - TITLE
reducer_2D_title = umap.UMAP(n_components=2, random_state=42, min_dist=0.3, spread=10.0)
reduced_title_emb = reducer_2D_title.fit_transform(title_embeddings)
# reduced_title_emb = reducer_2D_title.transform(title_embeddings)

plt.scatter(reduced_title_emb[:, 0],reduced_title_emb[:, 1], s=8, alpha=0.78, linewidths=0, color=c)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP', fontsize=24);

# ====== UMAP boiler plate ======
single_month = watch_data[watch_data['date'].dt.to_period('M') == '2025-04']
single_month_ids = single_month['id'].unique().tolist()
mask = title_embeddings.index.isin(single_month_ids)
masked_cats = categories.loc[mask]

n_components = 2
n_neighbors = 100
fit = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.0,
    n_components=n_components,
    metric='euclidean'
)
umapped = fit.fit_transform(title_embeddings);

umapped_month = u[mask]
month_colors = masked_cats.map(title_colours)

fig = plt.figure()
if n_components == 1:
    ax = fig.add_subplot(111)
    ax.scatter(umapped_month[:,0], range(len(umapped_month)), c=month_colors, s=5)
if n_components == 2:
    ax = fig.add_subplot(111)
    ax.scatter(umapped_month[:,0], umapped_month[:,1], c=month_colors, s=10, alpha=1)
if n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(umapped_month[:,0], umapped_month[:,1], u[:,2], c=month_colors, s=5)