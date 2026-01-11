#!/usr/bin/env python3


# Needed libraries 

# System requiremenmts 
import time
import random
import datetime as dt
import os

# Data Management 
import pandas as pd

# Web scrapping functions 
import web_minning as wm


### ---------------------------------------------------------------------------------------------
### Main Code -----------------------------------------------------------------------------------


# Parameters 
CITY = "cdmx"
TAGS_LIST = wm.city_hashtags(CITY)


RAW_DATA = "../data/raw/"
DIR_LOG = "../logs/"
FILE_LOG = "tiktok_city_minning_human_exp.log"
ROOT_FILE_ROOT = f"root_links_{CITY}_"
DAY_FILE = f"day_file_{CITY}_{pd.to_datetime(dt.datetime.today()).strftime('%Y-%m-%d')}"
DAYS_FRESHNESS = 300
IND_DELETE_SOURCE = False # Delete previous files used 

# Logger 

import logging

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
formatter = logging.Formatter(fmt)

console = logging.StreamHandler()
console.setFormatter(formatter)

file = logging.FileHandler(DIR_LOG + FILE_LOG)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)




### ---------------------------------------------------------------------------------------------
### Main Code -----------------------------------------------------------------------------------

if __name__=="__main__":




    # Get data 

    # Todays videos labeled as today and formatred date column 
    df.date_long = pd.to_datetime(df.date_long.apply(lambda x: 
                    pd.to_datetime(dt.datetime.today()).strftime('%Y-%m-%d') if 'Hace' in x else x)
                    , format='%Y-%m-%d')

    # Auxilliar string date limit 
    date_limit = pd.to_datetime(
        dt.datetime.today()- dt.timedelta(days=DAYS_FRESHNESS)
        ).strftime('%Y-%m-%d')

    # Maintain just 'recent' posts 
    df = df.query(f""" date_long > '{date_limit}'""")

    # Drop no description assets 
    df = df[~(df.description.isna() & df.long_description.isna())]

    # Format incomplete row descriptions 
    df['description'] = df['description'].fillna('')
    df['long_description'] = df['long_description'].fillna('')