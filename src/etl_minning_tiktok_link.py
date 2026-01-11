#!/usr/bin/env python3



# Needed libraries 

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.key_input import KeyInput
from selenium.webdriver.common.actions.pointer_input import PointerInput

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

    pending_files_list = os.listdir(RAW_DATA)
    pending_files_list = [x for x in pending_files_list if ROOT_FILE_ROOT in x]


    if pending_files_list != []:
        logger.info("In queue file list not empty empty. Link minning process continues.")

        for i,f in enumerate(pending_files_list):
            logger.info(f"Working with file {f} --- ")
            df_raw = pd.read_csv(RAW_DATA + f).drop_duplicates(subset=["href"])[:10]


            ### -----------------------------------------------------------------------------------------
            ### Content extraction ----------------------------------------------------------------------
        
            logger.info("Content extraction --- ")

            root_list = df_raw.href.to_list()
            caption_list = df_raw.description.to_list()

            desc_results = []

            for h, d in zip(root_list, caption_list):

                try:
                    # New driver window (more robust bot detection)
                    driver = webdriver.Chrome()
                    wait = WebDriverWait(driver, 15)
                    actions = ActionChains(driver)
                    keyboard = KeyInput("keyboard")
                    actions_b = ActionBuilder(driver, keyboard=keyboard)

                    driver.get(h)

                    time.sleep(random.uniform(2.5,3.5))

                    # Expand video description
                    more_buton = driver.find_element(
                        By.CSS_SELECTOR,
                        "button[class*='ButtonExpand']"
                    )

                    more_buton.click()

                    time.sleep(random.uniform(2.5,3.5))


                    # Date Extraction 
                    try:
                        raw = driver.find_element(
                        By.CSS_SELECTOR,
                        "span[class*='StyledTUXText']"
                        ).text
                        date_text = raw.replace("·","").replace(" ","")
                    except Exception:
                        date_text = ""

                    time.sleep(random.uniform(2.5,3.5))

                    # Description extraction
                    try:
                        raw = driver.find_element(
                        By.CSS_SELECTOR,
                        "div[class*='DivOverlayBottomContent']"
                        ).text

                        ia_disclaimer = "Esta información se generó por IA y puede presentar resultados que no son relevantes. No representa las opiniones o consejos de TikTok. Si tienes alguna duda, envíanosla a través de: Comentarios y ayuda: TikTok"

                        desc_text = raw.replace(ia_disclaimer,"")

                    except Exception:
                        desc_text = ""

                    time.sleep(random.uniform(0.5,1.5))


                    # Get Locations 

                    ind_found = False
                    lat = 0.0
                    lon = 0.0
                    osm_name_label = ""


                    try:

                        # Get address label

                        # From caption
                        add_label = driver.find_element(
                            By.CSS_SELECTOR,
                            "div[data-e2e='poi-tag'] p"
                        ).text.strip()


                        # Call API if we have label
                        if add_label != "":

                            loc_osm_api = wm.geocode_osm(add_label)
                            
                            # Update interest values
                            if loc_osm_api["found"]:
                                ind_found = True
                                lat = loc_osm_api["lat"]
                                lon = loc_osm_api["lon"]
                                osm_name_label = loc_osm_api["display_name"]

                        logger.info(f"Geolocation Process ended for {h}")
                            
                    # Empty values 
                    except:
                        logger.warning(f"Geolocation Process Failed for {h}")
                        continue 
                        

                    finally:
                        # Update results 
                        desc_results.append({
                                    "href": h,
                                    "long_description": desc_text,
                                    "date_long": date_text, 
                                    "add_label_used": add_label,
                                    "ind_found":ind_found, 
                                    "lat":lat,
                                    "lon":lon, 
                                    "osm_name_label":osm_name_label
                                                            })

                        time.sleep(random.uniform(0.5,1.5))


                        driver.quit()

                except:
                    # Force window closing 
                    driver.quit()


            logger.info("Hashtag Content Extraction --- ")

            logger.info("Saving root file --- ")

            df_raw_des = pd.DataFrame(desc_results)

            df = (df_raw.merge(
            df_raw_des
            , on = 'href', how = 'left'
            )
            .dropna(subset=['date_long'])
            )

            # Saving file 
            df.to_csv(RAW_DATA + DAY_FILE + f"_{str(i).zfill(5)}.csv", index = False)

            # Deleting old files 
            if IND_DELETE_SOURCE:
                os.remove(RAW_DATA + f)
                logger.info(f"File {f} removed.")

            logger.info(f"TikTok minning process ended for file {f}  --- ")