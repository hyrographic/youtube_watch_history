import pandas as pd
import numpy as np

from datetime import datetime

from pathlib import Path
import itertools
import time
from typing import Literal
import re
import lxml
from tqdm import tqdm
import json

from bs4 import BeautifulSoup as bs4
import yt_dlp

import seaborn as sns
from matplotlib import pyplot as plt

# ====== read data ======
def read_html_files(paths: list, save_json=None):
    history_data_list = []
    for fp in paths:
        resvoled = Path(fp).resolve()
        with open(resvoled, encoding='utf-8') as f:
            raw_history = f.read()
        history_soup = bs4(raw_history, 'lxml')
        history_data_list.append((fp, history_soup))

    outer_cells_dict = {fp:soup.find_all(attrs={'class':'outer-cell'}) for fp, soup in history_data_list}
    print('Number of outer cells: ', sum([len(v) for k,v in outer_cells_dict.items()]))

    history_json = []
    for fp, outer_cells in outer_cells_dict.items():
        for cell in tqdm(outer_cells):
            divs = cell.find_all('div')
            video_info = divs[0].find(attrs={'class':'content-cell'})
            video_info_dict = {re.search('watch|channel|post|playables', a['href'])[0] : {'title':a.get_text(), 'link':a['href']} for a in video_info.find_all('a')}
            strings = list(video_info.strings)
            date = strings[-1].replace('\u202f', ' ')
            date_parsed = datetime.strptime(date, '%b %d, %Y, %I:%M:%S %p %Z')
            action = strings[0].replace('\xa0', '')
            record_dict = {
                'date':date_parsed,
                'video_info': video_info_dict,
                'action':action,
                'file':fp
            }
            history_json.append(record_dict)
    print(f'{len(history_json)} records parsed')
    
    if save_json==None:
        save = input('SAVE TO JSON? Y/N')
    else:
        save = 'y'
    if save.lower() == 'y':    
        # save parsed data
        parsed_path = Path('data/watch_history_parsed.json').resolve()
        with open(parsed_path, 'w') as f:
            json.dump(history_json, f, default=str)
        print(f'saved to {parsed_path}')
    return

def read_parsed_data():
    parsed_path = Path('data/watch_history_parsed.json').resolve()
    with open(parsed_path, 'r') as f:
        history_json = json.load(f)
        print(f'Read {len(history_json)} lines of watch history data')
    
    history_df = pd.DataFrame(history_json)
    video_info_norm = pd.json_normalize(history_df['video_info']).rename(columns=lambda x: x.replace('.', '_'))
    watch_history_df = pd.concat([history_df, video_info_norm], axis=1)
    watch_history_df.drop(columns=['video_info'], inplace=True)
    watch_history_df['date'] = pd.to_datetime(watch_history_df['date'])
    
    # remove any duplicates across diff files
    duplicate_i = watch_history_df[watch_history_df.duplicated(subset=['date', 'watch_link'])].index
    watch_history_df.drop(index=duplicate_i, inplace=True)
    watch_history_df.reset_index(drop=True, inplace=True)

    # add id column
    watch_history_df['id'] = watch_history_df['watch_link'].apply(lambda x: np.nan if pd.isnull(x) else x.split('=')[-1])
    print(f'Dropped {len(duplicate_i)} duplicates')
    print(f'Number of rows remaining: {len(watch_history_df)}')
    return watch_history_df

def filter_by_media(df, selected_media: Literal['watch', 'playables', 'post']):
    media_types = ['watch', 'playables', 'post']
    media_types.pop(media_types.index(selected_media))
    non_cols = [c for c in df.columns if any([media_types[0] in c, media_types[1] in c]) & ('_' in c)]
    filtered = df.dropna(subset=[f'{selected_media}_link']).drop(columns=non_cols)
    return filtered

def filter_by_date(df, year, month, day):
    date_start = (year, month, day)
    aged_data_i = df[df['date'] < datetime(*date_start)].index
    df.drop(index=aged_data_i, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return

def get_yt_metadata(url):
    ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'skip_download': True,
    'cookies':str(Path('../cookies/www.youtube.com_cookies.txt').resolve())
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'id': info.get('id'),
                'title': info.get('title'),
                'description': info.get('description'),
                'channel': info.get('channel'),
                'channel_id': info.get('channel_id'),
                'upload_date': info.get('upload_date'),
                'timestamp': info.get('timestamp'),
                'duration': info.get('duration'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'comment_count': info.get('comment_count'),
                'categories': info.get('categories'),
                'tags': info.get('tags'),
                'media_type': info.get('media_type'),
                'channel_follower_count': info.get('channel_follower_count'),
                'channel_is_verified': info.get('channel_is_verified'),
                'availability': info.get('availability'),
                'age_limit': info.get('age_limit'),
                'live_status': info.get('live_status'),
                'thumbnail': info.get('thumbnail'),
                'url': url,
            }
    except Exception as e:
        return {'error': str(e), 'url': url}

def scrape_with_resume(urls, output_file='data/video_metadata.jsonl', delay=1):
    # Load already-processed URLs
    done = set()
    if Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    done.add(record.get('url'))
                except json.JSONDecodeError:
                    continue

    total = len(urls)
    remaining = [u for u in urls if u not in done]
    print(f'{len(done)} already done, {len(remaining)} remaining')

    with open(output_file, 'a') as f:
        for i, url in enumerate(remaining, 1):
            result = get_yt_metadata(url)
            f.write(json.dumps(result, default=str) + '\n')
            f.flush()
            if i % 100 == 0:
                print(f'Progress: {len(done) + i}/{total}')
            time.sleep(delay)

    print('Complete')

def read_metadata(paths: list):
    metadata_lines = []
    for path in paths:
        fp = Path(path).resolve()
        with open(fp) as f:
            metadata_lines.extend([json.loads(line) for line in f])
    metadata = pd.DataFrame(metadata_lines)
    print(f'Read {len(metadata)} lines of video metadata')
    return metadata

def metadata_group_errors(df):
    error_type_regex = (
        '(Sign in to confirm you’re not a bot|'
        'live stream recording is not available|'
        'available in your country|'
        'members-only content|'
        'violating YouTube|'
        'video is not available|'
        'Video unavailable|'
        'confirm your age|'
        'copyright claim|'
        'video has been terminated|'
        'Private video)'
    )

    df['error_type'] = df['error'].str.extract(error_type_regex)
    print('Error breakdown', df['error_type'].value_counts())
    print('\nUngrouped errors: ')
    print(df[df['error_type'].isna()]['error'].value_counts(dropna=False))
    return

#! ====== RUN ======
if __name__ == "main":
    # ====== read html data ======
    data_paths = [
        'data/watch-history-a1.html',
        'data/watch-history-a2.html',
        'data/watch-history-b1.html'
    ]
    read_html_files(paths=data_paths, save_json='y')
    # ====== read JSON data ======
    all_activities_df = read_parsed_data()

    # filter for watch data only
    non_watch_cols = [c for c in all_activities_df.columns if any(['playables' in c, 'post' in c]) & ('_' in c)]
    watch_history_df = all_activities_df.dropna(subset=['watch_link']).drop(columns=non_watch_cols)

    # filter for year where bulk of data is
    filter_by_date(watch_history_df, 2023, 1, 11)

    # get video meta data
    # urls = watch_history_df['watch_link'].dropna().unique().tolist()
    # scrape_with_resume(urls, output_file='data/video_metadata.jsonl', delay=0.5)

    # read video meta data
    metadata_paths = [
        'data/video_metadata.jsonl'
    ]
    metadata_df = read_metadata(metadata_paths)
    metadata_group_errors(metadata_df)

    # metadata_df.info()

    # ====== basic stats ======

    #? ====== timeseries ======
    gper = pd.Grouper('date', freq='D')
    watches_series = watch_history_df.set_index('date').groupby(gper)['watch_link'].nunique()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x=watches_series.index, height=watches_series.values, width=3)
    plt.xticks(rotation=90)
    plt.show()

    #? ====== time of day ======
    hr_ = watch_history_df['date'].dt.hour
    watches_by_tod = watch_history_df.groupby(hr_)['watch_link'].nunique()

    fig, ax = plt.subplots(figsize=(15, 5))
    watches_by_tod.plot.barh(x='date', y='watch_link')
    plt.xticks(rotation=90)
    plt.show()