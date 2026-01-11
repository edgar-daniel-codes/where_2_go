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
import web_minning as wm


### ---------------------------------------------------------------------------------------------
### Main Code -----------------------------------------------------------------------------------


# Parameters 
CITY = "cdmx"
TAGS_LIST = wm.city_hashtags(CITY)


RAW_DATA = "../data/raw/"
DIR_LOG = "../logs/"
FILE_LOG = "tiktok_city_minning_human_exp.log"
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
    df_raw.drop_duplicates(subset=['href'])
    df_raw.to_csv(RAW_DATA + ROOT_FILE, index = False)

    logger.info(f"TikTok exploration process ended for city {CITY} --- ")