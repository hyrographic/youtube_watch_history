import itertools
import pandas as pd
from datetime import datetime
import re
import json
from bs4 import BeautifulSoup as bs4
from pathlib import Path
import lxml
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# ====== read data ======
def read_html_files(paths: list):
    history_data_list = []
    for p in paths:
        resvoled = Path(p).resolve()
        with open(resvoled, encoding='utf-8') as f:
            raw_history = f.read()
        history_soup = bs4(raw_history, 'lxml')
        history_data_list.append(history_soup)

    outer_cells = list(itertools.chain(*[soup.find_all(attrs={'class':'outer-cell'}) for soup in history_data_list]))
    print('Number of outer cells: ', len(outer_cells))

    history_json = []
    for cell in tqdm(outer_cells):
        divs = cell.find_all('div')
        p = divs[0].find('p')
        video_info = divs[0].find(attrs={'class':'content-cell'})
        video_info_dict = {re.search('watch|channel|post|playables', a['href'])[0] : {'title':a.get_text(), 'link':a['href']} for a in video_info.find_all('a')}
        strings = list(video_info.strings)
        date = strings[-1].replace('\u202f', ' ')
        date_parsed = datetime.strptime(date, '%b %d, %Y, %I:%M:%S %p %Z')
        action = strings[0].replace('\xa0', '')
        record_dict = {
            'watch_date':date_parsed,
            'video_info': video_info_dict,
            'action':action
        }
        history_json.append(record_dict)
    print(f'{len(history_json)} records parsed')
    
    save = input('SAVE TO JSON? Y/N')
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
        print(f'Read {len(history_json)} lines')
    
    history_df = pd.DataFrame(history_json)
    video_info_norm = pd.json_normalize(history_df['video_info'])
    watch_history_df = pd.concat([history_df, video_info_norm], axis=1)
    watch_history_df.drop(columns=['video_info'], inplace=True)
    watch_history_df['watch_date'] = pd.to_datetime(watch_history_df['watch_date'])
    return watch_history_df    

#! ====== RUN ======
# ====== read html data ======
data_paths = [
    'data/watch-history.html'
]
# read_html_files(paths=data_paths)
# ====== read JSON data ======
watch_history_df = read_parsed_data()

# ====== stats ======

#? ====== timeseries ======
gper = pd.Grouper('watch_date', freq='W')
watches_weekly = watch_history_df.set_index('watch_date').groupby(gper)['watch.link'].nunique()
sns