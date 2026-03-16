# %%
# MARK: Imports
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
import math
import itertools

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
import plotly.graph_objects as go

# machine learning
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import umap

# nlp
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# local imports
from src import read_and_load_data

# MARK: Read Data
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

# remove any livestreams
livestreams = mdata_nlp[mdata_nlp['media_type'] == 'livestream'].index
mdata_nlp.drop(index=livestreams, inplace=True)

media_types_dict = mdata_nlp['media_type'].to_dict()

channels = mdata_nlp['channel'].fillna('None')

categories = mdata_nlp['categories'].apply(lambda d: d[0] if isinstance(d, list) else 'None')
categories.name = 'category'

title_data = mdata_nlp[mdata_nlp['title'].notna()]
titles = title_data['title']

desc_data = mdata_nlp[(mdata_nlp['description'].notna()) & (mdata_nlp['description'] != '')]
desc = desc_data['description']

tags_data = mdata_nlp[(mdata_nlp['tags'].notna()) &(mdata_nlp['tags'].apply(len) > 0)]
tags = tags_data['tags'].str.join(' ')

# MARK: NLP Setup
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

def create_transformer(model):
    return SentenceTransformer(model, trust_remote_code=True)

def encode_cached(m, transformer, values, file_name, index):
    if file_name:
        fp = f'data/embeddings_cache/{m}/{file_name}'
        os.makedirs('/'.join(fp.split('/')[:-1]), exist_ok=True)
        if os.path.exists(fp):
            print(f'Loading cached {file_name}')
            try:
                return pd.DataFrame(np.load(fp), index=index)
            except ValueError as e:
                print('Error:', e)
                print('re-encoding values')
                pass
    print(f'Encoding {file_name}...')
    emb = transformer.encode(values, batch_size=256, show_progress_bar=True)
    if file_name:
        np.save(fp, emb)
    return pd.DataFrame(emb, index=index)

def tokenise(text, number_tokens=Literal['keep', 'drop']):
    if not isinstance(text, str):
        return []
    allowed_chars = r"[a-z0-9']+"
    tokens = [w for w in re.findall(allowed_chars, text.lower()) if w not in stopwords_list and len(w) > 1]
    if number_tokens == 'drop':
        tokens = [w for w in tokens if not bool(re.match(r'^[0-9]+$', w))]
    return tokens

# MARK: Clean Tags
# ====== clean tags data ======
tag_values = tags_data['tags'].explode()

# get additional tags from description
description_hashtags = desc.str.extractall(r'(\#[\w]+\b)')
description_hashtags = description_hashtags.reset_index(level=1)[0].str.replace('#', '')
description_hashtags.dropna(inplace=True)

tag_values = pd.concat([tag_values, description_hashtags])

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
tag_values = tag_values[~tag_values.isin(stopwords_list)] # redo stop words as new noise might be created after removing "shorts"
# remove tags which reference the channel name
_channel_regex = r'((\’|\')s|[^a-z0-9])'
_channels_tmp = mdata_nlp['channel'].str.lower().str.strip().str.replace(_channel_regex, '', regex=1).fillna('')

# remove tag == channel name
tag_values = tag_values[tag_values.str.replace(_channel_regex, '', regex=1) != _channels_tmp.reindex(tag_values.index)]

# remove tags within channel name
tag_values = tag_values[[tag.replace(' ', '') not in channel for tag, channel in zip(tag_values, _channels_tmp.reindex(tag_values.index))]]

tag_values.dropna(inplace=True)

print('unique tags: ', tag_values.nunique())
tag_values.value_counts().iloc[50:].head(25)
# tag_values.value_counts(normalize=True).head(25)

# custom additions
custom_additions = []
def add_custom_tags(channel: str, tags_to_add: list):
    _ids = mdata_nlp[mdata_nlp['channel'] == channel].index.tolist()
    _to_add = pd.Series(
        [t for vid in _ids for t in tags_to_add],
        index=[vid for vid in _ids for t in tags_to_add]
        )
    custom_additions.append(_to_add)
    return _to_add

add_custom_tags('Zack D. Films', ['entertainment'])
add_custom_tags('Very Important People', ['comedy'])
add_custom_tags('JCS - Criminal Psychology', ['documentary'])
# add_custom_tags('Channel 5 with Andrew Callaghan', ['news'])
add_custom_tags('Adam Savage’s Tested', ['diy'])

tags_cleaned = pd.concat([tag_values, *custom_additions]).groupby(level=0).agg(list)

# MARK: Clean Titles
# ====== clean title data ======
title_values = title_data['title']
title_tokens = title_values.apply(tokenise, number_tokens='drop').explode()

title_tokens = (
    title_tokens
    .str.lower()
    .str.strip()
)

# apply string replacements
title_tokens = title_tokens.map(str_replacements, na_action=None).fillna(title_tokens)
# remove shorts
title_tokens = title_tokens.str.replace(' ?shorts ?', '', regex=1)

title_tokens.dropna(inplace=True)

print('unique words: ', title_tokens.nunique())
title_tokens.value_counts().head(25)
# title_tokens.value_counts(normalize=True).head(25)

titles_cleaned = title_tokens.groupby(level=0).agg(lambda x: ' '.join(x)).reindex(mdata_nlp.index).fillna(mdata_nlp['channel'])

# MARK: Clean Categories
# categories_clean = categories.str.replace('&', '', regex=1)
# categories_clean = categories_clean.str.replace(r'\s+', ' ', regex=1)
categories_clean = categories.str.title().str.strip()
# removes category entirely - ignored during reassignment as replaced with None
remove_categories = [
    'Nonprofits & Activism'
]
add_to_none = categories_clean.isin(remove_categories)
categories_clean[add_to_none] = 'None'

# MARK: Clean Description
# remove hashtags (included in tags already)
desc_values = desc.str.replace(r'(\#[\w]+\b)', '', regex=1)

# take first line only
desc_line = desc_values.str.split('\n').str[0].explode()

# drop any lines plugging socials
follow_strings = '^(subscribe|follow|Learn more|discover|insta|facebook|channel|website|watch here|watch on|click here|buy me|patreon|support|socials|more)'
desc_line = desc_line[~desc_line.str.contains(follow_strings, case=False)]

# remove links
url_regex = r'((https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
desc_line = desc_line.str.replace(url_regex, '', regex=1)

# remove tags
tag_regex = r'\@[\w\.\-\_]'
desc_line = desc_line.str.replace(tag_regex, '', regex=1)

# remove empty
desc_line = desc_line[desc_line != '']

desc_tokens = desc_line.apply(tokenise, number_tokens='drop')
desc_snippets = desc_tokens.str[:10].str.join(' ')

# desc_token_count = pd.Series(dict(Counter(list(itertools.chain(*desc_tokens.values)))))
# desc_token_count.sort_values(ascending=False).head(30)

# MARK: Embeddings
# %%
m = 'all-MiniLM-L6-v2' # fast model
# m = 'all-mpnet-base-v2' #slow model
# m = 'distilbert-base-nli-mean-tokens'
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# m = 'nomic-ai/nomic-embed-text-v1.5'
sent_transformer = create_transformer(m)

# ====== Embed on single combined string ======
_combined_idx = titles_cleaned.index.union(tags_cleaned.index)
_titles_aligned = titles_cleaned.reindex(_combined_idx)
_tags_aligned = tags_cleaned.reindex(_combined_idx)
_channels_aligned = channels.reindex(_combined_idx)

snippet_used_counter = 0

def compose(vid_id, include_tags=True, fill_w_desc=True, min_tokens=10):
    parts = []
    channel = _channels_aligned.loc[vid_id]
    if isinstance(channel, str) and channel:
        parts.append(channel)
    title = _titles_aligned.loc[vid_id]
    if isinstance(title, str) and title:
        parts.append(title)
    if include_tags:
        tag_list = _tags_aligned.loc[vid_id]
        if isinstance(tag_list, list) and tag_list:
            parts.append(' '.join(tag_list[:10]))
    composed = ' | '.join(parts)
    
    has_snippet = vid_id in desc_snippets
    under_min = len(composed.split()) < min_tokens
    if fill_w_desc and under_min and has_snippet:
        global snippet_used_counter
        snippet_used_counter += 1
        snippet = desc_snippets.loc[vid_id]
        if isinstance(snippet, str) and snippet:
            composed = f"{composed} | {snippet}"
    return composed

composed = pd.Series([compose(i, min_tokens=10_000) for i in _combined_idx], index=_combined_idx)
#.sample(round(len(_combined_idx)*0.3), random_state=5) # ! TESTING SAMPLE

composed_embeddings = encode_cached(m, sent_transformer, composed.tolist(), 'channel_title_tag_desc_forced.npy', composed.index)
composed_cats = categories.loc[composed.index].copy()

# MARK: Fix Categories
# %%
cat_m = 'all-MiniLM-L6-v2'
# cat_m = 'nomic-ai/nomic-embed-text-v1.5'
category_model = create_transformer(cat_m)

category_detail = {
    'People & Blogs': 'people blogs culture job career workplace',
    'Entertainment': 'entertainment',
    'Gaming': 'gaming overwatch gta minecraft',
    'Comedy': 'comedy skit sketch',
    'Science & Technology': 'science technology biology engineering',
    'Education': 'education history religion aviation law documentary how-to tutorial mathematics',
    'Music': 'music art edm',
    'Sports': 'sports football',
    'Film & Animation': 'film animation movie tv',
    'Autos & Vehicles': 'autos vehicles cars',
    'Travel & Events': 'travel adventure line-mission',
    'News & Politics': 'news politics events media',
    'Pets & Animals': 'pets animals ',
    'Makers & Hobby':'diy printing maker cad modelling electronics'
    # 'Howto & Style': 'howto tutorial cooking diy',
    # 'Coding, Data & Analysis':'data analysis tableau coding'
    # 'Cooking & Food':'cooking food'
}

category_rename = {
    'Music':'Music & Arts'
}

# composed category words embeddings
_cat_words = {
    key: ' '.join([w.replace('-', ' ') for w in re.split(r'[^a-zA-Z0-9\-]+', desc) if len(w) > 1])
    for key, desc in category_detail.items()
}
_word_emb_df = encode_cached(cat_m, category_model, list(_cat_words.values()), None, _cat_words)
category_word_embeddings = dict(zip(_word_emb_df.index, _word_emb_df.values))

# For each video, aggregate its tag embeddings and find nearest category
threshold = 0.05
updated_categories = []
for i, original_category in tqdm(categories_clean.items()):
    if isinstance(original_category, list):
        original_category = original_category[0]

    if i in composed_embeddings.index:
        composed_emb = composed_embeddings.loc[i]
    else:
        print(i)
        print('Not found!')
    
    norm_vid_emb = composed_emb / np.linalg.norm(composed_emb)

    sims = {}
    for cat, word_vecs in category_word_embeddings.items():
        norm_cat_words = word_vecs / np.linalg.norm(word_vecs, keepdims=True)
        score = np.dot(norm_cat_words, norm_vid_emb)
        # best_idx = int(scores.argmax())
        sims[cat] = {
            'score': score,
            'word': 'All composed'
        }


    
    og_cat_score = sims.get(original_category, {'score':0})['score']

    best_cat = max(sims, key=lambda x: sims.get(x)['score'])
    best_score = sims[best_cat]['score']
    best_cat_detail = sims[best_cat]['word']
    
    change_cat = (best_score - og_cat_score) > threshold

    updated_categories.append(
        {'id':i,
            'og_cat':category_rename.get(original_category, original_category),
            'og_cat_score':og_cat_score,
            'alt_cat':best_cat,
            'alt_cat_score':best_score,
            'alt_cat_detail_word':best_cat_detail,
            'changed': all([change_cat, best_cat != original_category]),
            'score_diff':f'{best_score - og_cat_score:+0.3f}',
            'final_cat': category_rename.get(best_cat, best_cat) if change_cat else category_rename.get(original_category, original_category)
            }
    )

adjusted_categories_df = pd.json_normalize(updated_categories).set_index('id')
adjusted_categories = adjusted_categories_df['final_cat']

# MARK: Category Analysis
# %% 
def category_change_summary(vid_id_set=None):
    if isinstance(vid_id_set, list):
        df = adjusted_categories_df[adjusted_categories_df.isin(vid_id_set)].copy()
    elif isinstance(vid_id_set, pd.Index) | isinstance(vid_id_set, pd.Series):
        df = adjusted_categories_df[adjusted_categories_df.index.isin(vid_id_set.tolist())].copy()
    else:
        df = adjusted_categories_df.copy()
        
    
    category_changed = df[df['changed'].fillna(False)]
    print(f'Total Number of categories changed: {len(category_changed):,}')

    summary = df['final_cat'].value_counts().sort_values(ascending=False).to_frame().rename(columns={'count':'resulting_cat_size'})

    summary['starting_cat_size'] = df['og_cat'].value_counts()
    summary['count_lost'] = category_changed['og_cat'].value_counts()
    summary['count_gained'] = category_changed['final_cat'].value_counts()

    fill_regex = r'((?:\+|\-)nan)'
    summary['- lost'] = summary['count_lost'].apply(lambda x: f'-{x:0}').str.replace(fill_regex, '--', regex=1)
    summary['+ gained'] = summary['count_gained'].apply(lambda x: f'+{x:0}').str.replace(fill_regex, '--', regex=1)
    
    summary['change'] = (summary['resulting_cat_size'] - summary['starting_cat_size']).divide(summary['starting_cat_size'])
    summary['percent_change'] = summary['change'].apply(lambda x: f'{x:+0.2%}')

    print(f'Proportion of each category changed')
    print(summary[['- lost', '+ gained', 'resulting_cat_size', 'percent_change']].sort_values('resulting_cat_size', ascending=False))
    return summary

category_change_summary();

#%%
# find random channels and tags
c1 = adjusted_categories_df['og_cat'] == 'Film & Animation'
c2 = adjusted_categories_df['alt_cat'] == 'Education'
selected_ids = adjusted_categories_df[c1 & c2].index.tolist()
selected_mdata = mdata_nlp[mdata_nlp.index.isin(selected_ids)]

selected_mdata['channel'].value_counts().head(25)

# selected_tags = list(itertools.chain(*tags_cleaned.loc[selected_ids].values))
# pd.Series(selected_tags).value_counts().head(30)

selected_channel = 'Vsauce'
channel_vid_ids = mdata_nlp[mdata_nlp['channel'] == selected_channel].index
category_change_summary(channel_vid_ids);

channel_tags = tags_cleaned.loc[[_ for _ in channel_vid_ids if _ in tags_cleaned]]
channel_selected_tags = list(itertools.chain(*channel_tags.values))
pd.Series(channel_selected_tags).value_counts().head(29)

selected_channel_adj_info = adjusted_categories_df.loc[channel_vid_ids]
selected_channel_adj_info[selected_channel_adj_info['final_cat'] == 'Film & Animation']
selected_channel_adj_info

# %%
# print tags per category
ids_per_cat = adjusted_categories_df.reset_index().groupby(['final_cat'])['id'].agg(list)
ids_per_cat.sort_values(key=lambda x: x.apply(len), ascending=False, inplace=True)
for _cat, ids in ids_per_cat.items():
    print(_cat, ':', len(ids), 'videos')
    tags_flat = list(itertools.chain(*tags_cleaned.loc[[_ for _ in ids if _ in tags_cleaned]].values))
    print(pd.Series(tags_flat).value_counts().head(10))
    print('\n')

# MARK: Category Sankey
# %%
# select custom df for sankey
changed_gby = (
    adjusted_categories_df
    # .loc[selected_ids]
    .groupby(['og_cat', 'final_cat'], as_index=False)
    .size()
)
# changed_gby['cat_change'] = changed_gby[['og_cat', 'final_cat']].apply(lambda x: '->'.join(x), axis=1)

og_filter_for = []
best_filter_for = []

mask = pd.Series(True, index=changed_gby.index)

if og_filter_for:
    mask &= changed_gby['og_cat'].isin(og_filter_for)
if best_filter_for:
    mask &= changed_gby['final_cat'].isin(best_filter_for)

_OTHER_THRESHOLD = 5

_sk_raw = changed_gby[mask].copy().sort_values('size', ascending=False)

# Paths with size < threshold are collapsed into a single "Other → Other" flow
_large = _sk_raw[_sk_raw['size'] >= _OTHER_THRESHOLD]
_small = _sk_raw[_sk_raw['size'] <  _OTHER_THRESHOLD]

_sk_df = _large.copy()
if not _small.empty:
    _sk_df = pd.concat([
        _sk_df,
        pd.DataFrame([{'og_cat': 'Other', 'final_cat': 'Other', 'size': _small['size'].sum()}]),
    ], ignore_index=True)
_sk_df = _sk_df.sort_values('size', ascending=False)

# Build node list: always suffix target names to prevent circular merging
_src_names  = list(_sk_df['og_cat'].unique())
_tgt_cats   = list(_sk_df['final_cat'].unique())
_tgt_names  = [n + '\u200b' for n in _tgt_cats]   # zero-width space = unique key, invisible in labels
_tgt_key    = {cat: cat + '\u200b' for cat in _tgt_cats}
_nodes      = _src_names + _tgt_names
_node_labels = _src_names + _tgt_cats              # clean display labels
_node_idx   = {n: i for i, n in enumerate(_nodes)}

# Assign colours from tab20 to source nodes; targets inherit from largest inflow
_cmap      = plt.cm.tab20
_src_rgb   = {n: tuple(int(c * 255) for c in _cmap(i % 20 / 20)[:3])
              for i, n in enumerate(_src_names)}

def _rgba(name, alpha=0.8):
    r, g, b = _src_rgb.get(name, (140, 140, 140))
    return f'rgba({r},{g},{b},{alpha})'

def _tgt_color(cat):
    rows = _sk_df[_sk_df['final_cat'] == cat]
    if rows.empty:
        return 'rgba(140,140,140,1.0)'
    top_src = rows.sort_values('size').iloc[-1]['og_cat']
    return _rgba(top_src, 1.0)

_node_colors = [
    _rgba(n, 1.0) if n in _src_rgb else _tgt_color(n.rstrip('\u200b'))
    for n in _nodes
]

_link_sources = [_node_idx[r['og_cat']]           for _, r in _sk_df.iterrows()]
_link_targets = [_node_idx[_tgt_key[r['final_cat']]] for _, r in _sk_df.iterrows()]
_link_values  = _sk_df['size'].tolist()
_link_colors  = [_rgba(r['og_cat'], 0.4) for _, r in _sk_df.iterrows()]
_link_labels  = [f"{r['og_cat']} → {r['final_cat']}: {r['size']:,}"
                 for _, r in _sk_df.iterrows()]

_fig_sk = go.Figure(go.Sankey(
    arrangement='snap',
    node=dict(
        label=_node_labels,
        color=_node_colors,
        pad=12,
        thickness=18,
        line=dict(width=0),
    ),
    link=dict(
        source=_link_sources,
        target=_link_targets,
        value=_link_values,
        color=_link_colors,
        label=_link_labels,
    ),
))
_fig_sk.update_layout(
    title='Category Reassignments',
    font=dict(size=11, color='white'),
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#0a0a0a',
    height=max(500, len(_src_names) * 30),
)
_fig_sk.show()

# MARK: PLOT FUNC
# ============== Plot ==============
# %%
def plot_circular_chart(
    umap_embeddings,
    umap_cats,
    sub_labels,
    watch_data,
    color_by='subcluster',      # 'subcluster' | 'channel' | 'category'
    shape_mode=True,
    highlight_ids=None,          # collection of video IDs to draw a ring around
    highlight_color='white',     # ring edge colour
    figsize=(14, 14),
    dpi=300,
    base_s=2.5,
    save_path=None,
    show=True,
):
    """
    Render the circular category chart.

    Each category's centroid is anchored on the unit-circle perimeter. Points are
    coloured and (optionally) shaped by sub-cluster, channel, or category. An
    optional highlight overlay draws a white ring around a subset of IDs.

    Returns
    -------
    fig         : Figure
    coords_df   : DataFrame (index=video_id, cols=[0,1]) — circular coordinates
    umap_colors : list — per-video colour, same order as umap_embeddings.index
    shape_marker: dict | None
    cat_order   : list
    cat_angles  : dict
    cat_colours : dict
    """
    import colorsys

    with open('resources/color_palette_50+.txt', 'r') as f:
        local_pal = f.read().split('\n')
    palette = sns.color_palette(local_pal)
    cat_colours = {cat: palette[i] for i, cat in enumerate(umap_cats.unique())}

    if color_by == 'subcluster':
        colour_map: dict = {}
        for cat in umap_cats.unique():
            base_rgb = cat_colours[cat]
            h, s, v = colorsys.rgb_to_hsv(*base_rgb)
            sub_ids = sorted(sub_labels[umap_cats == cat].unique())
            n_real = [x for x in sub_ids if x != -1]
            real_idx = {sub: i for i, sub in enumerate(n_real)}
            n = len(n_real)
            hue_spread = min(0.28, n * 0.06)
            for sub in sub_ids:
                if sub == -1:
                    colour_map[(cat, sub)] = colorsys.hsv_to_rgb(h, s * 0.25, v * 0.30)
                else:
                    t = real_idx[sub] / max(n - 1, 1)
                    new_h = (h + (t - 0.5) * hue_spread) % 1.0
                    new_s = float(np.clip(s * (0.6 + 0.8 * t), 0, 1))
                    new_v = 0.65 + 0.35 * t
                    colour_map[(cat, sub)] = colorsys.hsv_to_rgb(new_h, new_s, new_v)
        umap_colors = [colour_map[(cat, sub)] for cat, sub in zip(umap_cats, sub_labels)]
        shape_values = sub_labels.values

    elif color_by == 'channel':
        id_to_channel = watch_data.set_index('id')['channel_title'].to_dict()
        channels = pd.Series(umap_embeddings.index.map(id_to_channel), index=umap_embeddings.index).fillna('Unknown')
        top_channels = channels.value_counts().head(19).index.tolist()
        ch_palette = sns.color_palette('tab20', 20)
        ch_colours = {ch: ch_palette[i] for i, ch in enumerate(top_channels)}
        umap_colors = [ch_colours.get(ch, (0.45, 0.45, 0.45)) for ch in channels]
        shape_values = channels.values

    else:
        umap_colors = list(umap_cats.map(cat_colours))
        shape_values = umap_cats.values

    # ── Shape markers ─────────────────────────────────────────────────────────
    _markers = {'o': 1.0, 's': 0.9, '^': 1.2, 'D': 0.85, 'v': 1.2,
                'p': 0.9, 'h': 0.9, 'P': 0.85, '*': 3, 'X': 0.85}
    if shape_mode:
        unique_shape_vals = list(dict.fromkeys(shape_values))
        mkers_list = list(_markers.keys())
        shape_marker = {v: mkers_list[i % len(mkers_list)] for i, v in enumerate(unique_shape_vals)}
    else:
        shape_marker = None

    # ── Circular layout ────────────────────────────────────────────────────────
    cats_series = umap_cats.loc[umap_embeddings.index]
    cat_order = cats_series.value_counts().index.tolist()
    n_cats = len(cat_order)
    cat_angles = {cat: 2 * np.pi * i / n_cats for i, cat in enumerate(cat_order)}

    raw_centroids = {
        cat: umap_embeddings.loc[cats_series == cat].values.mean(axis=0)
        for cat in cat_order
    }

    coords_circ = np.empty_like(umap_embeddings.values, dtype=float)
    for cat in cat_order:
        idx = cats_series[cats_series == cat].index
        pts = umap_embeddings.loc[idx].values
        cent = raw_centroids[cat]
        target_angle = cat_angles[cat]
        target_r = 1

        rel = pts - cent
        raw_angle = np.arctan2(cent[1], cent[0])
        rot = target_angle - raw_angle
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rotated = np.column_stack([
            rel[:, 0] * cos_r - rel[:, 1] * sin_r,
            rel[:, 0] * sin_r + rel[:, 1] * cos_r,
        ])

        spread = np.abs(rel).max() or 1.0
        rotated /= spread * (2.4 / target_r)

        tx = target_r * np.cos(target_angle)
        ty = target_r * np.sin(target_angle)
        row_mask = umap_embeddings.index.get_indexer(idx)
        coords_circ[row_mask] = rotated + np.array([tx, ty])

    coords_df = pd.DataFrame(coords_circ, index=umap_embeddings.index, columns=[0, 1])

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')

    circle = plt.Circle((0, 0), 1.0, color='white', fill=False, lw=0.5, alpha=0.15)
    ax.add_patch(circle)

    c_arr = np.array(umap_colors, dtype=float)
    if shape_mode and shape_marker is not None:
        for val, marker in shape_marker.items():
            mask = shape_values == val
            ms = base_s * _markers.get(marker, 1.0)
            ax.scatter(coords_circ[mask, 0], coords_circ[mask, 1],
                       c=c_arr[mask], s=ms, alpha=0.5, linewidths=0, marker=marker)
    else:
        ax.scatter(coords_circ[:, 0], coords_circ[:, 1], c=c_arr, s=base_s, alpha=0.45, linewidths=0)

    # ── Optional highlight overlay (ring around selected points) ─────────────
    if highlight_ids is not None:
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon as MplPolygon
        h_coords = coords_df.loc[coords_df.index.isin(highlight_ids)].values
        # print(h_coords)
        if len(h_coords) >= 3:
            hull = ConvexHull(h_coords)
            hull_verts = h_coords[hull.vertices]
            poly = MplPolygon(hull_verts, closed=True, fill=False,
                              edgecolor=highlight_color, linewidth=0.8, alpha=0.7, zorder=5)
            ax.add_patch(poly)
        elif len(h_coords) == 2:
            ax.plot(h_coords[:, 0], h_coords[:, 1],
                    color=highlight_color, linewidth=2, alpha=0.7, zorder=5)

    # ── Category labels ───────────────────────────────────────────────────────
    for cat in cat_order:
        angle = cat_angles[cat]
        lx = 1.12 * np.cos(angle)
        ly = 1.12 * np.sin(angle)
        ha = 'left' if np.cos(angle) > 0.1 else ('right' if np.cos(angle) < -0.1 else 'center')
        ax.text(lx, ly, cat, color=cat_colours[cat], fontsize=7, ha=ha, va='center',
                fontweight='bold')

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis('off')

    handles = [Patch(color=cat_colours[cat], label=cat) for cat in cat_colours]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, frameon=False, labelcolor='white', fontsize=7)

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig, coords_df, umap_colors, shape_marker, cat_order, cat_angles, cat_colours
#%%

# MARK: UMAP / Cluster
#%%
class PerCategoryUMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=20, min_dist=0.15, spread=5.0, n_components=2, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.min_dist    = min_dist
        self.spread      = spread
        self.n_components = n_components
        self.metric      = metric

    def fit(self, X, y=None):
        self.categories_ = y  # y = per-video category Series
        return self

    def transform(self, X):
        umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            spread=self.spread,
            n_components=self.n_components,
            metric=self.metric,
        )
        result = pd.DataFrame(index=X.index, columns=range(self.n_components), dtype=float)
        for cat, idx in self.categories_.groupby(self.categories_).groups.items():
            if len(idx) < self.n_neighbors + 1:
                continue
            result.loc[idx] = umap_model.fit_transform(X.loc[idx])
        return result


class PerCategoryHDBSCAN(BaseEstimator):
    """Runs HDBSCAN per category; cluster size params scale with category size."""
    def __init__(self, min_cluster_size_frac=0.02, min_cluster_size_floor=5,
                 min_samples_frac=0.0001, min_samples_floor=3,
                 cluster_selection_epsilon=0.0, cluster_selection_method='leaf', allow_single_cluster=False):
        self.min_cluster_size_frac    = min_cluster_size_frac
        self.min_cluster_size_floor   = min_cluster_size_floor
        self.min_samples_frac         = min_samples_frac
        self.min_samples_floor        = min_samples_floor
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method  = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

    def _params_for(self, n_points):
        return dict(
            min_cluster_size=max(self.min_cluster_size_floor, int(n_points * self.min_cluster_size_frac)),
            min_samples=max(self.min_samples_floor, int(n_points * self.min_samples_frac)),
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster = self.allow_single_cluster,
            metric='euclidean',
            copy=True,
        )

    def fit_predict(self, X, y):  # y = per-video category Series
        labels = pd.Series(-1, index=X.index, name='sub_label', dtype=int)
        for cat, idx in y.groupby(y).groups.items():
            pts    = X.loc[idx].dropna()  # skip rows skipped by per-category UMAP
            if pts.empty:
                continue
            params = self._params_for(len(pts))
            model  = HDBSCAN(**params)
            if len(pts) < model.min_cluster_size:
                continue
            try:
                labels.loc[pts.index] = model.fit_predict(pts)
            except TypeError as e:
                print(e)
                # epsilon too large for this category, fall back to no epsilon
                fallback = HDBSCAN(**{**params, 'cluster_selection_epsilon': 0.0})
                labels.loc[pts.index] = fallback.fit_predict(pts)
        return labels

# Parameters

EMBEDDING_PARAMS = {
    'pca__n_components': 100,
    'umap__n_neighbors': 20,
    'umap__min_dist':    0.05,
    'umap__spread':      3.0,
    'umap__n_components': 2,
    'umap__metric':      'euclidean',
}

HDBSCAN_PARAMS = {
    'min_cluster_size_frac':    0.0005,
    'min_cluster_size_floor':   4,
    'min_samples_frac':         0.002,
    'min_samples_floor':        7,
    'cluster_selection_epsilon': 0.2,
    'allow_single_cluster':True,
    'cluster_selection_method': 'eom'
}

# Pipeline

def _normalize_df(X):
    return pd.DataFrame(normalize(X), index=X.index)

embedding_pipeline = Pipeline([
    ('pca',       PCA()),
    ('normalize', FunctionTransformer(_normalize_df)),
    ('umap',      PerCategoryUMAPTransformer()),
])
embedding_pipeline.set_params(**EMBEDDING_PARAMS)
embedding_pipeline.named_steps['pca'].set_output(transform="pandas")

# ohe channel as feature
ohe = OneHotEncoder()
svd = TruncatedSVD(n_components=25)
encoded_channels = ohe.fit_transform(channels.reindex(composed_embeddings.index).values.reshape(-1, 1))
reduced_channels = svd.fit_transform(encoded_channels)

composed_embeddings_channel_feat = pd.DataFrame(np.hstack([composed_embeddings, reduced_channels]), index=composed_embeddings.index)

print('PCA + Normalise + UMAP...')
umap_cats = adjusted_categories.loc[composed_embeddings_channel_feat.index].copy()
umap_embeddings = embedding_pipeline.fit_transform(composed_embeddings_channel_feat, y=umap_cats)
print('PCA + Normalise + UMAP ✔')

print('HDBSCAN...')
hdbscan_model = PerCategoryHDBSCAN(**HDBSCAN_PARAMS)
sub_labels = hdbscan_model.fit_predict(umap_embeddings, y=umap_cats)
print('HDBSCAN ✔')

# draw plot
COLOR_BY   = 'subcluster'
SHAPE_MODE = True

fig, coords_df, umap_colors, shape_marker, cat_order, cat_angles, cat_colours = plot_circular_chart(
    umap_embeddings, umap_cats, sub_labels, watch_data,
    color_by=COLOR_BY, shape_mode=SHAPE_MODE,
    save_path=f"charts/dev_samples/Development Sample ({round(len(umap_embeddings)/1000, 1)}k) {datetime.today().strftime('%d-%b %H')}.svg",
)

# MARK: Cluster Summary
#%%
def sample_links(ids, n=2):
    sampled = random.sample(ids, min(n, len(ids)))
    return '  '.join(f'https://youtube.com/watch?v={i}' for i in sampled)

# Select a category
selected_cat = 'Music & Arts'
selected_cat_index = umap_cats[umap_cats==selected_cat].index

category_sample = sub_labels.loc[selected_cat_index].to_frame()
category_sample['channel'] = category_sample.index.map(mdata_nlp['channel'])
print(f'{selected_cat} category: {category_sample['sub_label'].nunique()} sub-clusters')
print_out = (
    category_sample
    .reset_index()
    .groupby(['sub_label'], as_index=False)
    .agg({
            'index':lambda x: len(set(x)),
            'channel':lambda x: Counter(x.values).most_common(5)
        })
)
print(f"{'Sub-Cluster':<13}{'Videos':<13}{'Top Channels':<25}")
for i, row in print_out.sort_values('index', ascending=False).iterrows():
    _print = (
        f"{str(row['sub_label']).replace('-1', 'NOISE'):<13}"
        f"{row['index']:<13}"
        + '| '.join([f"({cnt}) {ch}" for ch, cnt in dict(row['channel']).items()])
    )
    print(_print)
#%%
#MARK: Inspection 
inspect_index = {}

selected_media = 'short'
# inspect_index['media_type'] = mdata_nlp[mdata_nlp['media_type']==selected_media].index

selected_cat = 'Music & Arts'
inspect_index['category'] = umap_cats[umap_cats==selected_cat].index

# Select a sub-cluster
selected_sub_cluster = 86
inspect_index['subcluster'] = sub_labels[sub_labels == selected_sub_cluster].index

# select a channel
selected_channel = 'SciManDan'
# inspect_index['channel'] = channels[channels == selected_channel].index

inspect_union = list(set.intersection(*[set(v) for v in inspect_index.values()]))
s_df = umap_embeddings.loc[inspect_union].copy()
s_ids = s_df.index.tolist()


s_watch_data = watch_data[watch_data['id'].isin(s_ids)].copy()
s_watch_data['media_type'] = s_watch_data['id'].map(mdata_nlp['media_type'])
s_watch_data['tags'] = s_watch_data['id'].map(mdata_nlp['tags'])
s_watch_data['categories'] = s_watch_data['id'].map(mdata_nlp['categories'])
s_watch_data['title'] = s_watch_data['id'].map(mdata_nlp['title'])
s_watch_data['title_tokens'] = s_watch_data['title'].apply(tokenise, number_tokens='drop')

print(f'Showing: {sub_labels.loc[inspect_union].nunique()} clusters')
print(sub_labels.loc[inspect_union].value_counts().head(5))

for k, v in inspect_index.items():
    print(f'{k}: {len(v)} videos')

print('Watched from: ', s_watch_data['date'].min(), ' to ', s_watch_data['date'].max())

# ── Weekly watch bar chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17.5, 5))

weekly = s_watch_data.groupby(['media_type', pd.Grouper(key='date', freq='W')])['id'].count().reset_index()

media_type_colors = {
    'short': '#e84118',
    'video': '#192a56',
    'livestream': '#4cd137'
}

sns.lineplot(data=weekly, x='date', y='id', hue='media_type', linewidth=4.5, palette=media_type_colors, alpha=0.8, ax=ax)

# pivot for easy access
weekly_pivot = weekly.pivot(index='date', columns='media_type', values='id').fillna(0)

# add count labels along bottom for videos
weekly_pivot_monthly = weekly_pivot.resample('ME').sum()

if 'video' in weekly_pivot_monthly:
    for date, val in weekly_pivot_monthly['video'].items():
        if val > 0:
            ax.text(date, -ax.get_ylim()[1] * 0.05, f'{int(val):,}',
                   ha='center', va='top', fontsize=11, fontweight='bold', 
                   color=media_type_colors['video'], rotation=45)

if 'short' in weekly_pivot_monthly:
    for date, val in weekly_pivot_monthly['short'].items():
        if val > 0:
            ax.text(date, ax.get_ylim()[1] * 0.95, f'{int(val):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold', 
                   color=media_type_colors['short'], rotation=45)

# make space above and below
ax.set_ylim(-ax.get_ylim()[1] * 0.25, ax.get_ylim()[1] * 1.15)

# set legend to underneath the chart
ax.legend(title='Media Type', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

ax.set_xlabel('Week')
ax.set_ylabel('Videos watched')
ax.set_title(f'weekly watch activity | category {selected_cat} | sub-cluster: {selected_sub_cluster}')
ax.grid(alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ── Cluster highlight scatter ────────────────────────────────────────────────
plot_circular_chart(
    umap_embeddings, umap_cats, sub_labels, watch_data,
    color_by=COLOR_BY, shape_mode=SHAPE_MODE,
    highlight_ids=s_ids,
)

# ── Top Vals  ────────────────────────────────────────────────────────────────
def _yt_link(video_id: str) -> str:
    return f"https://youtu.be/{video_id}"

def print_top_n(feature: str, n=25, breakdown='media_type', data=s_watch_data):
    clean = data[data[feature].notna() & data[breakdown].notna()]
    breakdown_vals = clean[breakdown].unique().tolist()
    
    media_gby = (
        clean.groupby(breakdown)[feature]
        .apply(lambda x: Counter(x.dropna()).most_common(n))
        .apply(lambda x: x + [('', 0)] * (n - len(x)))
    )
    
    top_feature_by_media = (
        media_gby
        .explode()
        .to_frame()
        .reset_index()
        .pivot_table(columns=[breakdown], aggfunc=list)
        .explode(breakdown_vals)
    )

    # Build lookup: (breakdown_val, feature_val) → example video id
    examples: dict[tuple, str] = {}
    for bv, grp in clean.groupby(breakdown):
        for fv, sub in grp.groupby(feature):
            examples[(bv, fv)] = sub['id'].sample(1).iloc[0]

    CELL = 55
    print(f'TOP {feature.replace('channel_title', 'channels').upper()}')
    print(''.join(f"{cl:<{CELL}}" for cl in top_feature_by_media.columns))
    
    for _, row in top_feature_by_media.iterrows():
        cells = []
        for bv in breakdown_vals:
            val, cnt = row.get(bv, ('', 0))
            if not val or cnt == 0:
                cells.append(f"{'':{ CELL}}")
                continue
            link = _yt_link(examples[(bv, val)]) if (bv, val) in examples else ''
            cells.append(f"{cnt:<2}| {str(val)[:26]:<26} {link:<23}")
        
        if any(c.strip() for c in cells):
            print(' '.join(cells))
    
    print('\n')

print_top_n('channel_title')

print_top_n('tags', data=s_watch_data.explode('tags').fillna('**no tags**'))

print_top_n('categories', data=s_watch_data.explode('categories').fillna('**no category**'))

print_top_n('title_tokens', data=s_watch_data.explode('title_tokens').dropna())

_ = composed.loc[s_ids].sample(min(len(s_ids), 10))
print('\n'.join(f"{idx} | {val}" for idx, val in _.items()))
#%%

# MARK: Animation
# %%
import animation_rendering
reload(animation_rendering)
from animation_rendering import UMAPAnimationRenderer

renderer = UMAPAnimationRenderer(
    fps=30,
    seconds_per_day=0.2,
    window_size=15,
    glow_size=1.5,
    scale_by_duration=True,
    duration_min_size=7,
    duration_max_size=13,
    core_point_size=2,
    glow_layers=UMAPAnimationRenderer().make_glow_layers(n_rings=12, max_size=7, max_alpha=0.2)
)

# ── Pre-compute shared inputs ──────────────────────────────────────────────────
coords_df, cat_order, cat_angles, _ = renderer.compute_circular_layout(umap_embeddings, umap_cats)

colors_series   = pd.Series(umap_colors, index=umap_embeddings.index)
is_noise        = (sub_labels == -1)
duration_series = pd.Series(
    umap_embeddings.index.map(mdata_nlp['duration']),
    index=umap_embeddings.index,
)

render_range = watch_data[
    (watch_data['date'].dt.to_period('D') >= '2025-01-01') &
    (watch_data['date'].dt.to_period('D') <= '2025-03-01')
].copy()
render_range = render_range[render_range['id'].isin(umap_embeddings.index)]

# ── Quick single-frame preview ─────────────────────────────────────────────────
renderer.sample_frame(
    render_range, coords_df, colors_series, is_noise,
    day_idx=72,
    save_path='charts/sample_frame.svg',
    marker_series=sub_labels,
    shape_marker=shape_marker,
    duration_series=duration_series,
    cat_colours=cat_colours,
    cat_angles=cat_angles,
    cat_order=cat_order,
)

renderer.frame_to_date(72)
# %%

#MARK: Full Render
renderer.render(
    df=render_range,
    coords_df=coords_df,
    colors_series=colors_series,
    is_noise_series=is_noise,
    output_dir='charts/frames_v11',
    marker_series=sub_labels,
    shape_marker=shape_marker,
    duration_series=duration_series,
    cat_colours=cat_colours,
    cat_angles=cat_angles,
    cat_order=cat_order,
)