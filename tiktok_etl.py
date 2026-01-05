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

# Data Management 
import pandas as pd

# Web scrapping functions 
import src.web_minning as wm


### ---------------------------------------------------------------------------------------------
### Main Code -----------------------------------------------------------------------------------


# Parameters 
CITY = "cdmx"
TAGS_LIST = wm.city_hashtags(CITY)


RAW_DATA = "./data/raw/"
DIR_LOG = "./logs/"
FILE_LOG = "tiktok_city_minning.log"
ROOT_FILE = f"root_links_{CITY}_{pd.to_datetime(dt.datetime.today()).strftime('%Y-%m-%d')}.csv"
DAY_FILE = f"day_file_{CITY}_{pd.to_datetime(dt.datetime.today()).strftime('%Y-%m-%d')}.csv"
DAYS_FRESHNESS = 300

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
    
    ### -----------------------------------------------------------------------------------------
    ### Hashtag exploration -------------------------------------------------------------------------------
  

    logger.info("Start Hashtag Exploration --- ")

    results = []


    for ht in TAGS_LIST:


        try:
            # Invoke driver object 

            driver = webdriver.Chrome()
            wait = WebDriverWait(driver, 15)
            actions = ActionChains(driver)
            keyboard = KeyInput("keyboard")
            actions_b = ActionBuilder(driver, keyboard=keyboard)


            # Open Main page
            driver.get("https://www.tiktok.com")


            # Look for search icon 
            search_icon = wait.until(EC.presence_of_element_located((
                By.XPATH, "//*[@role='searchbox']"
            )))

            # Put mose in search bar 
            search_icon.click()

            # Input labels 
            actions.send_keys(ht)
            time.sleep(random.uniform(0.05, 0.15))
            actions.perform()

            # Enter to look 
            actions.send_keys("\n")
            actions.perform()

            time.sleep(random.uniform(1.5, 4.5))


            # Getting all inmediate results 
            items = wm.scroll_until_no_new_items(driver)
            wm.scroll_through_elements(driver, items)
            items = driver.find_elements(By.CSS_SELECTOR, "div[id^='grid-item-container-']")

            # Get data from results

            for container in items:
                try:
                    href = container.find_element(
                        By.CSS_SELECTOR, "a[href*='/video/']"
                    ).get_attribute("href")

                    description = container.find_element(
                        By.CSS_SELECTOR, "[data-e2e='new-desc-span']"
                    ).text.strip()

                    date = container.find_element(
                        By.CSS_SELECTOR, "div[class*='DivTimeTag']"
                    ).text.strip()

                    results.append({
                        "href": href,
                        "description": description,
                        "date": date,
                        "hashtag_label": ht
                    })

                except Exception:
                    continue

            logger.info(f"Links search for {ht} ended.")

            driver.quit()

        except:
            logger.warning(f"Links search for {ht} failed.")
            driver.quit()

    logger.info("Hashtag Exploration Ends --- ")

    logger.info("Saving root file --- ")
    df_raw = pd.DataFrame(results)
    df_raw['level'] = 0
    df_raw.to_csv(RAW_DATA + ROOT_FILE, index = False)




    ### -----------------------------------------------------------------------------------------
    ### Content extraction ----------------------------------------------------------------------
  
    logger.info("Content extraction --- ")

    root_list = df_raw.href.drop_duplicates().to_list()
    caption_list = df_raw.description.drop_duplicates().to_list()

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
                "div[class*='DivCustomTDKContainer']"
                ).text

                ia_disclaimer = "Esta información se generó por IA y puede presentar resultados que no son relevantes. No representa las opiniones o consejos de TikTok. Si tienes alguna duda, envíanosla a través de: Comentarios y ayuda: TikTok"

                desc_text = raw.replace(ia_disclaimer,"")

            except Exception:
                desc_text = ""

            time.sleep(random.uniform(0.5,1.5))


            # Get Locations 

            
            try:

                # Get address label

                # From caption 
                if d !='':
                    excat_add =  wm.extract_addresses(d, top_k=1, min_score=2.5)
                    if len(excat_add) > 0:
                        add_label =  excat_add[0].text
                    else:
                        if len(wm.detect_places(d))>0:
                            add_label =  wm.detect_places(d)[0].text
                        else:
                            add_label = ''

                # Fromn text description 
                else:
                    if desc_text != "":
                        excat_add =  wm.extract_addresses(desc_text, top_k=1, min_score=2.5)
                        if len(excat_add) > 0:
                            add_label =  excat_add[0].text
                        else:
                            if len(wm.detect_places(desc_text))>0:
                                add_label =  wm.detect_places(d)[0].text
                            else:
                                add_label = ''

                    else:
                        add_label = ""

                # Call API if we have label
                if add_label != "":

                    loc_osm_api = wm.geocode_osm(add_label)
                    
                    # Update interest values
                    if loc_osm_api["found"]:
                        ind_found = True
                        lat = loc_osm_api["lat"]
                        lon = loc_osm_api["lon"]
                        osm_name_label = loc_osm_api["display_name"]
                        

                    else:
                        ind_found = False
                        lat = 0.0
                        lon = 0.0
                        osm_name_label = ""

                # Empty values 
                else: 
                    ### [COMPLETE WITH SPEECH TO TEXT OR VIDEO TO TEXT]

                    ind_found = False
                    lat = 0.0
                    lon = 0.0
                    osm_name_label = ""

                logger.info(f"Geolocation Process ended for {h}")
                    
            # Empty values 
            except:


                ind_found = False
                lat = 0.0
                lon = 0.0
                osm_name_label = ""
                logger.warning(f"Geolocation Process Failed for {h}")

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

    # Saving file 
    df.to_csv(RAW_DATA + DAY_FILE, index = False)

    logger.info(f"TikTok minning process ended for city {CITY} --- ")