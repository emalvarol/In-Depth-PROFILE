# Script objective
'''
Web Scraping Amazon
Python 3.12.5 script to retreive Amazon item prices for economic
valuation. Because “Python is perfect for easy implementation” (Khder, 2021, p. 164)
'''

# region Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from playwright.sync_api import sync_playwright # Module for browser automation
from bs4 import BeautifulSoup  # Module for HTML parsing
import time # Module for time-related functions (sleep time for web scrapping to prevent from account blockage)                                             # sleep
from thefuzz import fuzz
import User_Paths_and_Inputs

#endregion

# region 0 --- Configuration for Input/Output Files ---
# region 0 Prepare configuration and data
# Set workspace and general variables
workspace = User_Paths_and_Inputs.workspace
PATHS = User_Paths_and_Inputs.PATHS
CODES = User_Paths_and_Inputs.CODES
DATA_PATH_PRICES = PATHS['prices_content']

# Set specific web scrapping configuration
WEB_SOURCE = {
    'Amazon': ['ENS', 'ELD', 'BEC', 'MOB', 'APE', 'DEC', 'INS',
               'OCI', 'ORD', 'ENE', 'OTR', 'HER', 'LIM', 'ESP', 'JOY',
               'ROP', 'HSB'],
    'Wallapop': ['VEH']
}

### Load data
## Survey
df = pd.read_excel(PATHS['survey'], sheet_name="Data")

def transform_survey_to_long(df):
    # Rename cold names with codes
    df = df.rename(columns={v: k for k, v in CODES['Survey'].items()})
    df['GRP'] = df['GRP'].map({v: k for k, v in CODES['Content'].items()}).fillna(df['GRP'])
    # Transform survey data to long format
    df_long = pd.melt(
        df,
        id_vars=list(CODES['Survey'].keys()),
        value_vars=[col for col in df.columns if col.startswith('V') and col[1:].isdigit()],
        var_name='SU',
        value_name='VAL'
    )
    df_long['SU'] = df_long['SU'].str[1:].astype(int)
    bt_map = df_long[df_long['ATR'] == 'BT'].set_index('SU')['VAL']
    df_long['BT'] = df_long['SU'].map(bt_map).astype(int)
    cols = ['SU', 'BT'] + [c for c in df_long.columns if c not in ['SU', 'BT', 'VAL']] + ['VAL']
    df_long = df_long[cols]
    return df_long
df_long = transform_survey_to_long(df)

# Prices df
# Create a new DataFrame with the required structure
prices_df = pd.DataFrame(columns=[
    'GRP',
    'ITM',
    'Scraped_Name',
    'Scraped_Price',
    'Scraped_Link',
    'Scraped_Date'
])
#endregion

# region 1 - Web Scraping
# Helping functions
def build_amazon_search_url(search_keyword, page=1, base_url="https://www.amazon.es"):
    """
    Builds an Amazon search URL for a given keyword and page number.
    """
    keyword_param = search_keyword.replace(" ", "+")
    url = f"{base_url}/s?k={keyword_param}"
    if page > 1:
        url += f"&page={page}"
    return url

def build_amazon_item_url(relative_link, base_url="https://www.amazon.es"):
    """
    Returns the full URL of the item. 
    If the link is already absolute, returns it as is.
    """
    if relative_link.startswith("http"):
        return relative_link
    return base_url.rstrip("/") + "/" + relative_link.lstrip("/")

def build_wallapop_search_url(search_keyword, min_price=None, max_price=None, conditions_str=None, category_id=None, base_url="https://es.wallapop.com/search?"):
    """
    Builds a Wallapop search URL based on query parameters.
    """
    params = []
    
    # 1. Price Range
    if min_price is not None:
        params.append(f"min_sale_price={min_price}")
    if max_price is not None:
        params.append(f"max_sale_price={max_price}")

    # 2. Keywords
    params.append("keywords=" + search_keyword.replace(" ", "+"))
    
    # 3. Category ID
    if category_id is not None:
        params.append(f"category_id={category_id}")
    
    # 4. Order and Source (Standard Wallapop defaults)
    params.append("order_by=most_relevance")
    params.append("source=side_bar_filters")

    # 5. Conditions
    if conditions_str:
        # URL encode the comma separator as %2C
        encoded_conditions = conditions_str.replace(",", "%2C")
        params.append(f"condition={encoded_conditions}")

    return base_url + "&".join(params)

def build_wallapop_item_url(relative_link, base_url="https://es.wallapop.com"):
    """
    Returns the full URL of the Wallapop item.
    """
    if relative_link.startswith("http"):
        return relative_link
    return base_url.rstrip("/") + "/" + relative_link.lstrip("/")

# region 1.1 - Inspecting the page / Web Analysis
# region 1.1.1 URL
# Unique Searchs
df_search = df_long[df_long['DOM'] == 'Contenido'][['GRP','ITM']].drop_duplicates().reset_index(drop=True)

# Get first query search to test everything
item_name = df_search['ITM'][0]

# Build url
url = build_amazon_search_url(item_name)

# Fetching (curl, wget)
'''
Fetchign does not work with Amazon because of dynamic
Java Script. So we make the fetching directly managing the
HTML dynamicly
'''
#endregion

# region 1.1.2 HTML Document  (dynamic)
'''
Playwright cannot be included into a class neither a function.
'''
# Start Playwright (returns a Playwright controller)
playwright = sync_playwright().start()
# Launch Chromium (use headless=True for headless mode)
browser = playwright.chromium.launch(headless=False)
# Create a new browser context (isolated cookies/storage)
context = browser.new_context()
# Open a new page (analogous to a Selenium 'driver' window)
page = context.new_page()
# Open the item webpage
_ = page.goto(url, wait_until="domcontentloaded")
# Reject cookies (Playwright uses CSS selectors directly)
page.click("#sp-cc-rejectall-link")
#endregion

# region 1.2 - Extract the data / Extraction
# Get a list of the items to be scraped
amazon_items_df = df_search[df_search['Code'].isin(WEB_SOURCE['Amazon'])]

# Main bucle
'''
Include crawler and scraper into a bucle to extract the info
from all the "items_list". Save it directly into prices_df
'''
# --- Configuration & Debugging ---
break_at = float('inf') # For debugging purposes (set float('inf') to run all)
break_count = 0 # For debugging purposes
stop_all = False # For debugging purposes
debug_break_code = None # For debugging purposes (set None to disable) 
MAX_PAGES_LIMIT = 1  # Restrict how many pages maximum to scrape per item
for idx, row in amazon_items_df.iterrows():
    # Debugging: Break condition by count
    break_count += 1
    if break_count > break_at:
        print(f"Reached break_at limit ({break_at}). Stopping.")
        break
    # Get item details
    group_name = row['06_Group']
    item_name = row['07_Item']
    group_code = row['Code']

    # Debugging: Break condition by specific group
    if group_name == debug_break_code:
        print(f"Manual break triggered at Group: {group_name}. Stopping.")
        break

    print(f"Scraping {break_count}/{len(amazon_items_df)}: [{group_code}] {item_name}")
    try:
        # region 1.2.1 Website Crawling (regular expressions, XPath, HTML parsing libraries)
        # Go to first page
        initial_url = build_amazon_search_url(item_name)
        _ = page.goto(initial_url, wait_until="domcontentloaded")

        # Identify how many pages exist
        try:
            pagination_items = page.locator('span.s-pagination-item')
            # Check if pagination exists
            if pagination_items.count() > 0:
                total_pages_text = pagination_items.nth(-1).inner_text()
                total_pages = int(total_pages_text)
                # Apply limits
                max_pages = min(total_pages, MAX_PAGES_LIMIT)
            else:
                max_pages = 1
        except Exception:
            max_pages = 1  # If no pagination, just one page
        
        # Crawling loop with tqdm
        all_soups = []
        for page_num in range(1, max_pages + 1):
            search_url = build_amazon_search_url(item_name, page=page_num)
            _ = page.goto(search_url, wait_until="domcontentloaded")
            html = page.content()
            soup = BeautifulSoup(html, "lxml")
            all_soups.append(soup)
        #print(f"Collected HTML for {len(all_soups)} pages.")


        #endregion
        # region 1.2.2 Scraper
        # Scraping loop with tqdm
        scraped_results = []
        for soup in tqdm(all_soups, desc=f"Scraping items"):
            # Find all item containers
            item_cards = soup.find_all("div", attrs={"role": "listitem"})

            for card in item_cards:
                # Title and link
                link_tag = card.select_one('a.a-link-normal.s-link-style.a-text-normal')
                if link_tag:
                    link = build_amazon_item_url(link_tag.get("href"))
                    title_span = link_tag.select_one('h2 span')
                    title = title_span.get_text(strip=True) if title_span else None
                else:
                    link = None
                    title = None

                # Price
                price_tag = card.select_one('span.a-offscreen')
                if price_tag:
                    price_text = price_tag.get_text(strip=True)
                    price_text = price_text.replace('€', '').replace('\xa0', '').replace(',', '.')
                    try:
                        price = float(price_text)
                    except ValueError:
                        price = None
                else:
                    price = None

                # Add to local list if valid
                if title and price and link:
                    scraped_results.append({
                        "06_Group": group_name,
                        "Group_Code": group_code,
                        "07_Item": item_name,
                        "Scraped_Name": title,
                        "Scraped_Price": price,
                        "Scraped_Link": link,
                        "Scraped_Date": pd.Timestamp.now()
                    })

        # Remove duplicates by title and items with null price
        unique_results_dict = {res['Scraped_Name']: res for res in scraped_results}
        final_items = list(unique_results_dict.values())

        # Print items
        #print(len(items)) # e.g.: 120

        # Save items into excell
        if final_items:
            # Append new data to prices_df
            new_rows_df = pd.DataFrame(final_items)
            prices_df = pd.concat([prices_df, new_rows_df], ignore_index=True)
            
            # Save to disk incrementally
            prices_df.to_excel(DATA_PATH_PRICES, index=False)
            print(f"\tSuccessfully saved {len(final_items)} items.")
        else:
            print(f"\tNo valid data found for {item_name}")

    except Exception as e:
        print(f"Error processing {item_name}: {e}")
        continue

#endregion

# region 1.1bis - Inspecting the page / Web Analysis
# region 1.1.1bis URL

# Build url
url = build_wallapop_search_url(item_name)

# Fetching (curl, wget)
'''
Fetchign does not work with Amazon because of dynamic
Java Script. So we make the fetching directly managing the
HTML dynamicly
'''
#endregion

# region 1.1.2bis HTML Document  (dynamic)
'''
Playwright cannot be included into a class neither a function.
'''
# Start Playwright (returns a Playwright controller)
#playwright = sync_playwright().start()
# Launch Chromium (use headless=True for headless mode)
#browser = playwright.chromium.launch(headless=False)
# Create a new browser context (isolated cookies/storage)
#context = browser.new_context()
# Open a new page (analogous to a Selenium 'driver' window)
#page = context.new_page()
# Open the item webpage
_ = page.goto(url, wait_until="domcontentloaded")
# Reject cookies (Playwright uses CSS selectors directly)
page.click("#onetrust-reject-all-handler")
#endregion

# MANUAL!
'''
Here the Ubicación/Location is selected after first search
general_location = "Spain" # Ubicación
'''

# Main bucle
'''
Include crawler and scraper into a bucle to extract the info
from all the "items_list". Save it directly into prices_df
'''
# region 1.2bis - Extract the data / Extraction
# Get a list of the items to be scraped
wallapop_items_df = df_search[df_search['Code'].isin(WEB_SOURCE['Wallapop'])]
# Mapping based on wallapop_items_df index
# 100: Coches, 14000: Motos, 12579: Bicicletas/Patinetes
INDEX_TO_WALLAPOP_CAT = {
    65: 12579,   # Bicicleta
    99: 12579,   # Bicicletas Btwin aluminio
    145: 100,    # Coche o moto (Defaulted to car)
    172: 12579,  # Bicicleta estática
    296: 12579,  # Bicicleta Rockrider
    375: 100,    # Coche o moto - Fiat 1
    381: 14000,  # Moto
    382: 14000,  # Moto Thypoon
    400: 14000,  # Coche o moto - Moto 125cc
    412: 100,    # Coche o moto - Avor A3
    424: 12579,  # Patinete eléctrico
    509: 12579,  # Bicicleta - Soomerence
    579: 100     # Coche o moto - Audi A1
}
ITEM_NAME_CLEANER = {
    145: "Coche",
    375: "Coche Fiat 1",
    400: "Moto 125cc",
    412: "Coche Audi A3",
    579: "Coche Audi A1"
}

# --- Configuration & Debugging ---
break_at = float('inf') # For debugging purposes (set float('inf') to run all)
break_count = 0 # For debugging purposes
stop_all = False # For debugging purposes
debug_break_code = None # For debugging purposes (set None to disable) 
MAX_SCROLL_LIMIT = 0  # Restrict how many pages maximum to scrape per item (NOT AVAILABLE YET)
TARGET_CONDITION = "new"
for idx, row in wallapop_items_df.iterrows():
    # Debugging: Break condition by count
    break_count += 1
    if break_count > break_at:
        print(f"Reached break_at limit ({break_at}). Stopping.")
        break
    # Get item details
    group_name = row['06_Group']
    item_name = row['07_Item']
    item_name = ITEM_NAME_CLEANER.get(idx, item_name)
    group_code = row['Code']

    # Debugging: Break condition by specific group
    if group_name == debug_break_code:
        print(f"Manual break triggered at Group: {group_name}. Stopping.")
        break
    
    # Retrieve the category ID using the current DataFrame index
    current_cat_id = INDEX_TO_WALLAPOP_CAT.get(idx)
    print(f"Scraping {break_count}/{len(wallapop_items_df)}: [{group_code}] {item_name} (Category: {current_cat_id})")
    try:
        # region 1.2.1 Website Crawling (regular expressions, XPath, HTML parsing libraries)
        # Go to first page
        initial_url = build_wallapop_search_url(
            item_name, 
            conditions_str=TARGET_CONDITION,
            category_id=current_cat_id
        )
        _ = page.goto(initial_url, wait_until="domcontentloaded")
        # Implementation of retry logic because of Wallapop loading issues
        success = False
        for attempt in range(3):
            try:
                # Navigate with long timeout
                page.goto(initial_url, timeout=60000, wait_until="domcontentloaded")
                
                # Wait for the main item cards to appear
                page.wait_for_selector("div.item-card_ItemCard__content__C1mfJ", timeout=5000)
                success = True
                break
            except Exception as e:
                print(f"\tTimeout or selector error on attempt {attempt + 1}: {item_name}. Retrying...")
                time.sleep(2)
        
        if not success:
            print(f"\tFailed to load Wallapop items for {item_name} after 3 attempts.")
            continue # Skip to next item in the main loop

        # Crawling loop
        all_soups = []
        #Scroll not available yet, it would be here
        # Capture the current state of the page
        html = page.content()
        soup = BeautifulSoup(html, "lxml")
        all_soups.append(soup)
        #print(f"Collected HTML for {len(all_soups)} pages.")


        #endregion
        # region 1.2.2 Scraper
        # Scraping loop with tqdm
        scraped_results = []
        for soup in tqdm(all_soups, desc=f"Scraping items"):
            # Use the selector from your reference class
            item_cards = soup.find_all("div", class_="item-card_ItemCard__content__C1mfJ")

            for card in item_cards:
                # Title
                title_tag = card.find("h3", class_="item-card_ItemCard__title__5TocV")
                title = title_tag.text.strip() if title_tag else None

                # Price
                price_tag = card.find("strong", class_="item-card_ItemCard__price__pVpdc")
                if price_tag:
                    # Logic from your reference: replace \xa0€ and handle decimals
                    price_text = price_tag.text.strip().replace("\xa0€", "")
                    try:
                        price = float(price_text)
                    except ValueError:
                        price = None
                else:
                    price = None

                # Link
                parent_a = card.find_parent("a") or card.select_one("a")
                relative_link = parent_a["href"] if parent_a else None
                link = build_wallapop_item_url(relative_link)

                # Add to local list if valid
                if title and price and link:
                    scraped_results.append({
                        "06_Group": group_name,
                        "Group_Code": group_code,
                        "07_Item": item_name,
                        "Scraped_Name": title,
                        "Scraped_Price": price,
                        "Scraped_Link": link,
                        "Scraped_Date": pd.Timestamp.now()
                    })

        # Remove duplicates by link and items with null price
        unique_results_dict = {res['Scraped_Link']: res for res in scraped_results}
        final_items = list(unique_results_dict.values())

        # Print items
        #print(len(items)) # e.g.: 120

        # Save items into excell
        if final_items:
            # Append new data to prices_df
            new_rows_df = pd.DataFrame(final_items)
            prices_df = pd.concat([prices_df, new_rows_df], ignore_index=True)
            
            # Save to disk incrementally
            prices_df.to_excel(DATA_PATH_PRICES, index=False)
            print(f"\tSuccessfully saved {len(final_items)} items.")
        else:
            print(f"\tNo valid data found for {item_name}")

    except Exception as e:
        print(f"Error processing {item_name}: {e}")
        continue



# region 1.3 - Store the data / Data Organization
# region 1.3.1 Desired Data
'''
Did it in the previous step, object "items"
'''

#endregion

# region 1.3.2 Transformation (Structured data)
# Cleaning and filtering
# use Large Lenguage AI Model (LLM)


#endregion


# region 1.3.3 Structured data (save the data)
# Close playwright and all classes

#endregion

#endregion

#endregion

# region 2 - Data Analysis
# 2.1 Create a common df to work with
# Using the prices_df generated in Region 1
df_cleaning = prices_df.copy()

# 2.2 Cleaning process
# 2.2.1 Remove any duplicated link
if not df_cleaning.empty:
    df_cleaning.drop_duplicates(subset=['Scraped_Link'], keep='first', inplace=True)

# 2.2.2 Remove high similar titles
def calculate_title_similarity(titles):
    if len(titles) < 2:
        return 100.0
    pairs = list(combinations(titles, 2))
    similarity_scores = [fuzz.ratio(p[0], p[1]) for p in pairs]
    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 100.0

if not df_cleaning.empty:
    ids_to_remove = []
    # Group by specific item search term and price
    grouped = df_cleaning.groupby(['ITM', 'Scraped_Price'])

    for name, group in grouped:
        if len(group) > 1:
            similarity = calculate_title_similarity(group['Scraped_Name'].dropna().tolist())
            if similarity > 80:
                # Keep the first, discard the rest
                ids_to_remove.extend(group.index.tolist()[1:])

    ids_to_remove = list(set(ids_to_remove))
    df_cleaning = df_cleaning.drop(index=ids_to_remove)

# 2.2.3 Remove non-sense item prices by setting price-range for some "items"
price_filters = {
    "AAPortatil": {"min_p": [50], "max_p": []},
    "Alfombras": {"min_p": [5], "max_p": []},
    "Antiguedades": {"min_p": [5], "max_p": []},
    "Aspiradora": {"min_p": [20], "max_p": []},
    "Bandejas": {"min_p": [5], "max_p": []},
    "Barbacoa": {"min_p": [20], "max_p": []},
    "Basura": {"min_p": [5], "max_p": []},
    "Batidora": {"min_p": [5], "max_p": []},
    "Baul": {"min_p": [20], "max_p": []},
    "Bebidas": {"min_p": [], "max_p": []},
    "Bicicletas": {"min_p": [50], "max_p": []},
    "Cafetera": {"min_p": [15], "max_p": []},
    "Camas": {"min_p": [50], "max_p": []},
    "Cepillo": {"min_p": [1], "max_p": [100]},
    "Chimenea": {"min_p": [], "max_p": []},
    "Cintas": {"min_p": [], "max_p": []},
    "Coleccion": {"min_p": [], "max_p": []},
    "Comida": {"min_p": [], "max_p": [100]},
    "Congelador": {"min_p": [20], "max_p": []},
    "Congeladores": {"min_p": [20], "max_p": []},
    "Cortinas": {"min_p": [5], "max_p": []},
    "Coser": {"min_p": [], "max_p": []},
    "Cristaleria": {"min_p": [5], "max_p": []},
    "Cuadros": {"min_p": [5], "max_p": []},
    "Cuberteria": {"min_p": [5], "max_p": []},
    "Cuchillos": {"min_p": [], "max_p": []},
    "DVD": {"min_p": [], "max_p": []},
    "Deporte": {"min_p": [], "max_p": []},
    "Deportivo": {"min_p": [], "max_p": []},
    "Electrodomesticos": {"min_p": [15], "max_p": []},
    "Eqmusica": {"min_p": [10], "max_p": []},
    "Especias": {"min_p": [], "max_p": [100]},
    "Espejos": {"min_p": [5], "max_p": []},
    "Figuras": {"min_p": [], "max_p": []},
    "Freidora": {"min_p": [15], "max_p": []},
    "Frigorifico": {"min_p": [50], "max_p": []},
    "HElectricas": {"min_p": [], "max_p": []},
    "Herramientas": {"min_p": [], "max_p": []},
    "Higiene": {"min_p": [], "max_p": []},
    "Horno": {"min_p": [50], "max_p": []},
    "Instmusic": {"min_p": [10], "max_p": []},
    "Jardineria": {"min_p": [5], "max_p": []},
    "Joyas": {"min_p": [10], "max_p": []},
    "Juguetes": {"min_p": [5], "max_p": []},
    "LampMesa": {"min_p": [10], "max_p": []},
    "LampPie": {"min_p": [10], "max_p": []},
    "Lavadora": {"min_p": [50], "max_p": []},
    "Lavavajillas": {"min_p": [50], "max_p": []},
    "Libros": {"min_p": [1], "max_p": []},
    "Limpieza": {"min_p": [1], "max_p": []},
    "Manteles": {"min_p": [1], "max_p": []},
    "Medicinas": {"min_p": [], "max_p": []},
    "Menaje": {"min_p": [], "max_p": []},
    "Mesa": {"min_p": [10], "max_p": []},
    "Mesas": {"min_p": [10], "max_p": []},
    "Microondas": {"min_p": [20], "max_p": []},
    "Muebles": {"min_p": [10], "max_p": []},
    "Nevera": {"min_p": [20], "max_p": []},
    "Ollas": {"min_p": [5], "max_p": []},
    "Ordenador": {"min_p": [50], "max_p": []},
    "Otros": {"min_p": [], "max_p": []},
    "Papelera": {"min_p": [1], "max_p": []},
    "Perfumes": {"min_p": [5], "max_p": []},
    "Portafotos": {"min_p": [1], "max_p": []},
    "Productos": {"min_p": [], "max_p": []},
    "Relojes": {"min_p": [10], "max_p": []},
    "Robot": {"min_p": [15], "max_p": []},
    "Ropa": {"min_p": [5], "max_p": [50]},
    "RopaCama": {"min_p": [5], "max_p": []},
    "Sartenes": {"min_p": [5], "max_p": []},
    "Secador": {"min_p": [5], "max_p": []},
    "Secadora": {"min_p": [15], "max_p": []},
    "Sillas": {"min_p": [10], "max_p": []},
    "Sofas": {"min_p": [20], "max_p": []},
    "Tabla": {"min_p": [10], "max_p": []},
    "Telefono": {"min_p": [15], "max_p": []},
    "Televisiones": {"min_p": [20], "max_p": []},
    "Tendedero": {"min_p": [], "max_p": []},
    "Toallas": {"min_p": [], "max_p": []},
    "Toalleros": {"min_p": [], "max_p": []},
    "Tostadora": {"min_p": [15], "max_p": []},
    "Trabajo": {"min_p": [], "max_p": []},
    "Utensilios": {"min_p": [], "max_p": []},
    "Vajillas": {"min_p": [], "max_p": []},
    "Vehiculo": {"min_p": [500], "max_p": []},
    "Ventiladores": {"min_p": [15], "max_p": []},
    "Videoconsolas": {"min_p": [20], "max_p": []},
    "Vitroceramica": {"min_p": [20], "max_p": []}
}

if not df_cleaning.empty:
    rows_to_keep = pd.Series(True, index=df_cleaning.index)

    for item, filters in price_filters.items():
        item_mask = (df_cleaning['GRP'] == item)
        
        min_p_list = filters.get('min_p', [])
        if min_p_list:
            min_price = min_p_list[0]
            rows_to_keep[item_mask & (df_cleaning['Scraped_Price'] < min_price)] = False
            
        max_p_list = filters.get('max_p', [])
        if max_p_list:
            max_price = max_p_list[0]
            rows_to_keep[item_mask & (df_cleaning['Scraped_Price'] > max_price)] = False

    df_cleaning = df_cleaning[rows_to_keep]

# 2.3 Update validated dataframe
if not prices_df.empty:
    prices_df['validated'] = 0
    # Map back indices from cleaned subset
    prices_df.loc[prices_df.index.isin(df_cleaning.index), 'validated'] = 1
    
    # Save the updated data directly to disk
    prices_df.to_excel(DATA_PATH_PRICES, index=False)

# endregion