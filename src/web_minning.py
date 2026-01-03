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

# Data Management 
import pandas as pd


def scroll_until_no_new_items(
    driver,
    item_selector="div[id^='grid-item-container-']",
    pause=1.5,
    max_rounds=30
):
    last_count = 0

    for i in range(max_rounds):
        items = driver.find_elements(By.CSS_SELECTOR, item_selector)
        current_count = len(items)

        if current_count == last_count:
            break  # no new content loaded

        last_count = current_count

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        time.sleep(pause)

    return driver.find_elements(By.CSS_SELECTOR, item_selector)

def scroll_through_elements(driver, elements, pause=0.5):
    for el in elements:
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'})", el
        )
        time.sleep(pause)


    
def city_hashtags(city):
    city = city.lower()

    base = [
        f"#dondeir{city}",
        f"#quehacer{city}",
        f"#lugares{city}",
        f"#places #{city}",
        f"#places{city}",
        f"#turismo{city}",
    ]

    discovery = [
        f"#lugaresbonitos #{city}",
        f"#lugaressecretos #{city}",
        f"#lugaresimperdibles #{city}",
        f"#lugaresrecomendados #{city}",
        f"#experiencias #{city}",
        f"#experienciasunicas #{city}",
        f"#lugaresbonitos{city}",
        f"#lugaressecretos{city}",
    ]

    time_intent = [
        f"#findesemana #{city}",
        f"#findesemana{city}",
        f"#planesdefin #{city}",
        f"#planperfecto  #{city}",
        f"#escapada{city}",
    ]

    food_coffee = [
        f"#brunch #{city}",
        f"#cafes #{city}",
        f"#restaurantes #{city}",
        f"#brunch{city}",
        f"#cafes{city}",
        f"#comida{city}",
        f"#restaurantes{city}",
    ]

    all_tags = (
        base
        + discovery
        + time_intent
        + food_coffee
    )

    # Deduplicate while preserving order
    seen = set()
    tags = [t for t in all_tags if not (t in seen or seen.add(t))]

    return tags
