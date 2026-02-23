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

# local imports
from importlib import reload
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

watch_data.assign(m=watch_data['date'].dt.to_period('M')).groupby(['m', 'file'])['watch_link'].nunique().sort_values().to_dict()

#? ====== time of day ======
# hr_ = watch_data['date'].dt.hour
# watches_by_tod = watch_data.groupby(hr_)['watch_link'].nunique()

# fig, ax = plt.subplots(figsize=(15, 5))
# watches_by_tod.plot.barh(x='date', y='watch_link')
# plt.xticks(rotation=90)
# plt.show()

# fig, ax = plt.subplots(figsize=(22, 3))
# ax.plot(watches_series.index, watches_series_smoothed, lw=1.5)
# # x axis
# plt.xticks(rotation=35)

#* ====== FONT SETTINGS ======
yt_sans_bold = Path('resources/fonts/YouTubeSansBold.otf').resolve()
font_manager.fontManager.addfont(yt_sans_bold)
LABEL = {'family': 'YouTube Sans', 'size': 14, 'weight':'bold'}

#* ====== END ======


#? ====== line timeseries ======
def plot_timeline(colour=Literal['grey', 'red'], save=None):
    gper = pd.Grouper('date', freq='W')
    watches_series = watch_data.set_index('date').groupby(gper)['watch_link'].nunique()

    ws_smoothed = pd.Series(
        gaussian_filter1d(watches_series.fillna(0), sigma=2.5),
        index = watches_series.index
        )

    fig, ax = plt.subplots(figsize=(20, 1.5), dpi=300)
    fig.tight_layout()
    
    # Break the line at each half year boundary
    half_years = pd.date_range(start=ws_smoothed.index.min(), end=ws_smoothed.index.max(), freq='6MS')

    for i, boundary in enumerate(half_years[:-1]):
        mask = (ws_smoothed.index >= boundary) & (ws_smoothed.index < half_years[i+1])
        segment = ws_smoothed[mask].copy()
        colour_hex = {'red':'#FE0000', 'grey':'#8E8F93'}.get(colour)
        ax.plot(segment.index, segment.values, lw=4, c=colour_hex)

    # x axis
    plt.xticks(rotation=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set(**LABEL)
        label.set_ha('left')
    ax.tick_params(axis='x', length=0)

    # Offset first and last labels inward
    fig.canvas.draw()
    labels = ax.get_xticklabels()
    offset = matplotlib.transforms.ScaledTranslation(10/25, 0, fig.dpi_scale_trans)
    labels[0].set_transform(labels[0].get_transform() + offset)
    offset = matplotlib.transforms.ScaledTranslation(-10/25, 0, fig.dpi_scale_trans)
    labels[-1].set_transform(labels[-1].get_transform() + offset)

    # y axis
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.show()
    if save:
        fp = Path(save).resolve()
        fig.savefig(fp, dpi=300, transparent=True)

plot_timeline(colour='red')
plot_timeline(colour='grey')

plot_timeline(colour='red', save='charts/red_timeline.png')
plot_timeline(colour='grey', save='charts/grey_timeline.png')