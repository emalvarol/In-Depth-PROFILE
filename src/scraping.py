"""
In-Depth-PROFILE Web Scraping Module
Handles automated extraction of economic asset data and market pricing.
"""
import time
import pandas as pd
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from itertools import combinations
from thefuzz import fuzz

class EconomicScraper:
    """
    Encapsulates web scraping workflows for market asset pricing
    used in the flood damage economic valuation engine.
    """
    def __init__(self, config_paths, config_codes):
        """
        Initializes the scraper with centralized framework paths and codes.
        Include configuration for all web sources
        
        Args:
            config_paths (dict): The PATHS dictionary imported from src.config
            config_codes (dict): The CODES dictionary imported from src.config
        """
        self.paths = config_paths
        self.codes = config_codes
        self.data_path_prices = self.paths['prices_content']
        
        # Specific web scraping configuration
        self.web_source = {
            'Amazon': [
                # Content items (excluding VEH)
                'APP', 'CLO', 'COM', 'DEC', 'ELE', 'ENG', 'FAD', 'FUR', 
                'HHG', 'HHB', 'INS', 'LEI', 'OTH', 'SPE', 'TOO',
                # Continent components
                'PUM', 'CLE', 'DHU', 'SOI', 'FRI', 'SKT', 'RDR', 'WND', 
                'PLG', 'PRW', 'EXF', 'ETP', 'ITP', 'ELS'
            ],
            'Wallapop': ['VEH']
        }
        self.max_pages_limit = 1
        
        # Wallapop specific configurations
        self.target_condition = "new"
        self.index_to_wallapop_cat = {
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
        self.item_name_cleaner = {
            145: "Coche",
            375: "Coche Fiat 1",
            400: "Moto 125cc",
            412: "Coche Audi A3",
            579: "Coche Audi A1"
        }
    
    # Amazon preparation
    @staticmethod
    def _build_amazon_search_url(search_keyword, page=1, base_url="https://www.amazon.es"):
        keyword_param = search_keyword.replace(" ", "+")
        url = f"{base_url}/s?k={keyword_param}"
        if page > 1:
            url += f"&page={page}"
        return url

    @staticmethod
    def _build_amazon_item_url(relative_link, base_url="https://www.amazon.es"):
        if relative_link.startswith("http"):
            return relative_link
        return base_url.rstrip("/") + "/" + relative_link.lstrip("/")

    # Wallapop preparation
    @staticmethod
    def _build_wallapop_search_url(search_keyword, min_price=None, max_price=None, conditions_str=None, category_id=None, base_url="https://es.wallapop.com/search?"):
        params = []
        if min_price is not None:
            params.append(f"min_sale_price={min_price}")
        if max_price is not None:
            params.append(f"max_sale_price={max_price}")
        params.append("keywords=" + search_keyword.replace(" ", "+"))
        if category_id is not None:
            params.append(f"category_id={category_id}")
        params.append("order_by=most_relevance")
        params.append("source=side_bar_filters")
        if conditions_str:
            encoded_conditions = conditions_str.replace(",", "%2C")
            params.append(f"condition={encoded_conditions}")
        return base_url + "&".join(params)
    
    @staticmethod
    def _build_wallapop_item_url(relative_link, base_url="https://es.wallapop.com"):
        if relative_link.startswith("http"):
            return relative_link
        return base_url.rstrip("/") + "/" + relative_link.lstrip("/")
    
    def _prepare_search_data(self):
        """
        Transforms survey data into the required search DataFrame.
        """
        df = pd.read_excel(self.paths['survey'], sheet_name="Data")
        df = df.rename(columns={v: k for k, v in self.codes['Survey'].items()})
        df['GRP'] = df['GRP'].map({v: k for k, v in self.codes['Content'].items()}).fillna(df['GRP'])
        
        df_long = pd.melt(
            df,
            id_vars=list(self.codes['Survey'].keys()),
            value_vars=[col for col in df.columns if col.startswith('V') and col[1:].isdigit()],
            var_name='SU',
            value_name='VAL'
        )
        df_long['SU'] = df_long['SU'].str[1:].astype(int)
        
        # Ensure Code, Group, and Item mappings match the expected input structure
        # Assumes the melted DataFrame contains 'DOM', 'GRP', 'ITM', and 'Code' equivalents
        df_search = df_long[df_long['DOM'] == 'Contenido'][['GRP', 'ITM']].drop_duplicates().reset_index(drop=True)
        
        # Rename for standard internal processing based on your old code references
        df_search = df_search.rename(columns={'GRP': 'Code', 'ITM': '07_Item'})
        df_search['06_Group'] = df_search['Code'] 
        
        return df_search

    def _extract_raw_prices_source_a(self, amazon_items_df):
        """
        Extracts raw data from Amazon using Playwright and BeautifulSoup.
        
        Args:
            amazon_items_df (pd.DataFrame): Filtered DataFrame containing items to scrape.
            
        Returns:
            pd.DataFrame: Accumulated scraped data.
        """
        print(f"Starting Amazon extraction for {len(amazon_items_df)} items...")
        
        prices_df = pd.DataFrame(columns=[
            'GRP', 'ITM', 'Scraped_Name', 'Scraped_Price', 'Scraped_Link', 'Scraped_Date'
        ])

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            
            # Initial dummy navigation to reject cookies
            test_url = self._build_amazon_search_url("test")
            page.goto(test_url, wait_until="domcontentloaded")
            try:
                page.click("#sp-cc-rejectall-link", timeout=3000)
            except Exception:
                pass

            break_count = 0
            for idx, row in amazon_items_df.iterrows():
                break_count += 1
                group_name = row['06_Group']
                item_name = row['07_Item']
                group_code = row['Code']

                print(f"Scraping {break_count}/{len(amazon_items_df)}: [{group_code}] {item_name}")
                
                try:
                    initial_url = self._build_amazon_search_url(item_name)
                    page.goto(initial_url, wait_until="domcontentloaded")

                    # Pagination check
                    try:
                        pagination_items = page.locator('span.s-pagination-item')
                        if pagination_items.count() > 0:
                            total_pages = int(pagination_items.nth(-1).inner_text())
                            max_pages = min(total_pages, self.max_pages_limit)
                        else:
                            max_pages = 1
                    except Exception:
                        max_pages = 1
                    
                    all_soups = []
                    for page_num in range(1, max_pages + 1):
                        search_url = self._build_amazon_search_url(item_name, page=page_num)
                        page.goto(search_url, wait_until="domcontentloaded")
                        soup = BeautifulSoup(page.content(), "lxml")
                        all_soups.append(soup)

                    scraped_results = []
                    for soup in tqdm(all_soups, desc="Scraping items", leave=False):
                        item_cards = soup.find_all("div", attrs={"role": "listitem"})

                        for card in item_cards:
                            link_tag = card.select_one('a.a-link-normal.s-link-style.a-text-normal')
                            if link_tag:
                                link = self._build_amazon_item_url(link_tag.get("href"))
                                title_span = link_tag.select_one('h2 span')
                                title = title_span.get_text(strip=True) if title_span else None
                            else:
                                link, title = None, None

                            price_tag = card.select_one('span.a-offscreen')
                            if price_tag:
                                price_text = price_tag.get_text(strip=True).replace('€', '').replace('\xa0', '').replace(',', '.')
                                try:
                                    price = float(price_text)
                                except ValueError:
                                    price = None
                            else:
                                price = None

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

                    # Remove duplicates by title
                    unique_results_dict = {res['Scraped_Name']: res for res in scraped_results}
                    final_items = list(unique_results_dict.values())

                    if final_items:
                        new_rows_df = pd.DataFrame(final_items)
                        prices_df = pd.concat([prices_df, new_rows_df], ignore_index=True)
                        prices_df.to_excel(self.data_path_prices, index=False)
                        print(f"\tSuccessfully saved {len(final_items)} items.")
                    else:
                        print(f"\tNo valid data found for {item_name}")

                except Exception as e:
                    print(f"Error processing {item_name}: {e}")
                    continue

            browser.close()
        
        return prices_df

    def _extract_raw_prices_source_b(self, wallapop_items_df, prices_df):
        """
        Extracts raw data from Wallapop using Playwright and BeautifulSoup.
        Handles manual user intervention for location selection on the first load.
        """
        print(f"Starting Wallapop extraction for {len(wallapop_items_df)} items...")
        
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False) # Headless=False es obligatorio para interactuar
            context = browser.new_context()
            page = context.new_page()
            
            # 1. Cargar primera búsqueda para forzar la configuración inicial
            if not wallapop_items_df.empty:
                first_row = wallapop_items_df.iloc[0]
                first_item = self.item_name_cleaner.get(wallapop_items_df.index[0], first_row['07_Item'])
                first_cat = self.index_to_wallapop_cat.get(wallapop_items_df.index[0])
                
                setup_url = self._build_wallapop_search_url(
                    first_item, conditions_str=self.target_condition, category_id=first_cat
                )
                page.goto(setup_url, wait_until="domcontentloaded")
                
                # Rechazo automático de cookies inicial
                try:
                    page.click("#onetrust-reject-all-handler", timeout=3000)
                except Exception:
                    pass
                
                # ==============================================================================
                # INTERVENCIÓN MANUAL DEL USUARIO
                # ==============================================================================
                print("\n[MANUAL INTERVENTION REQUIRED]")
                print("1. Go to the Chromium browser window.")
                print("1b. If cookies has not been yet accepted, accept them.")
                print("2. Set the desired Location/Ubicación (e.g., 'Spain' or local region).")
                print("3. Return to this terminal and press ENTER to start the automation...")
                input("Press ENTER to continue...") 
                # ==============================================================================

            # 2. Bucle principal de scraping (reutiliza la sesión con la ubicación guardada en cookies)
            break_count = 0
            for idx, row in wallapop_items_df.iterrows():
                break_count += 1
                group_name = row['06_Group']
                item_name = row['07_Item']
                group_code = row['Code']

                item_name = self.item_name_cleaner.get(idx, item_name)
                current_cat_id = self.index_to_wallapop_cat.get(idx)

                print(f"Scraping {break_count}/{len(wallapop_items_df)}: [{group_code}] {item_name}")
                
                try:
                    initial_url = self._build_wallapop_search_url(
                        item_name, 
                        conditions_str=self.target_condition,
                        category_id=current_cat_id
                    )
                    
                    success = False
                    for attempt in range(3):
                        try:
                            page.goto(initial_url, timeout=60000, wait_until="domcontentloaded")
                            page.wait_for_selector("div.item-card_ItemCard__content__C1mfJ", timeout=5000)
                            success = True
                            break
                        except Exception:
                            print(f"\tTimeout or selector error on attempt {attempt + 1}: {item_name}. Retrying...")
                            time.sleep(2)
                    
                    if not success:
                        print(f"\tFailed to load Wallapop items for {item_name} after 3 attempts.")
                        continue 

                    all_soups = [BeautifulSoup(page.content(), "lxml")]
                    scraped_results = []
                    
                    for soup in tqdm(all_soups, desc="Scraping items", leave=False):
                        item_cards = soup.find_all("div", class_="item-card_ItemCard__content__C1mfJ")

                        for card in item_cards:
                            title_tag = card.find("h3", class_="item-card_ItemCard__title__5TocV")
                            title = title_tag.text.strip() if title_tag else None

                            price_tag = card.find("strong", class_="item-card_ItemCard__price__pVpdc")
                            if price_tag:
                                price_text = price_tag.text.strip().replace("\xa0€", "")
                                try:
                                    price = float(price_text)
                                except ValueError:
                                    price = None
                            else:
                                price = None

                            parent_a = card.find_parent("a") or card.select_one("a")
                            relative_link = parent_a["href"] if parent_a else None
                            link = self._build_wallapop_item_url(relative_link) if relative_link else None

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

                    unique_results_dict = {res['Scraped_Link']: res for res in scraped_results}
                    final_items = list(unique_results_dict.values())

                    if final_items:
                        new_rows_df = pd.DataFrame(final_items)
                        prices_df = pd.concat([prices_df, new_rows_df], ignore_index=True)
                        prices_df.to_excel(self.data_path_prices, index=False)
                        print(f"\tSuccessfully saved {len(final_items)} items.")
                    else:
                        print(f"\tNo valid data found for {item_name}")

                except Exception as e:
                    print(f"Error processing {item_name}: {e}")
                    continue

            browser.close()
        
        return prices_df

    def run(self, extract_amazon=True, extract_wallapop=True):
        """
        Main public interface execution thread.
        """
        print("Starting Economic Asset Price Scraping Pipeline...")
        
        # Prepare search datasets
        print("Preparing search datasets...")
        df_search = self._prepare_search_data()
        amazon_items_df = df_search[df_search['Code'].isin(self.web_source['Amazon'])]
        wallapop_items_df = df_search[df_search['Code'].isin(self.web_source['Wallapop'])]
        
        # Execute extraction
        if not extract_amazon and not extract_wallapop:
            print("Data preparation successful. Skipping extraction as requested.")
            return df_search, amazon_items_df, wallapop_items_df
        
        if extract_amazon:
            print("Scrapping Amazon...")
            prices_df = self._extract_raw_prices_source_a(amazon_items_df)
        else:
            print("Skipping Amazon extraction phase.")
        
        if extract_wallapop:
            print("Scrapping Wallapop...")
            prices_df = self._extract_raw_prices_source_b(wallapop_items_df, prices_df)
        else:
            print("Skipping Wallapop extraction phase.")
        
        print("Scraping pipeline completed successfully.")

class PriceDataCleaner:
    """
    Validates and cleans scraped economic data by removing duplicates, 
    filtering by fuzzy title similarity, and applying price range thresholds.
    """
    def __init__(self, config_paths):
        self.paths = config_paths
        self.data_path_prices = self.paths['prices_content']
        
        # Price thresholds mapping
        self.price_filters = {
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

    @staticmethod
    def _calculate_title_similarity(titles):
        if len(titles) < 2:
            return 100.0
        pairs = list(combinations(titles, 2))
        similarity_scores = [fuzz.ratio(p[0], p[1]) for p in pairs]
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 100.0

    def clean_data(self):
        print("Starting data validation and cleaning process...")
        try:
            prices_df = pd.read_excel(self.data_path_prices)
        except FileNotFoundError:
            print("No scraped data file found to clean.")
            return

        if prices_df.empty:
            print("The dataset is empty.")
            return prices_df

        df_cleaning = prices_df.copy()
        
        # 1. Remove duplicate links
        df_cleaning.drop_duplicates(subset=['Scraped_Link'], keep='first', inplace=True)

        # 2. Remove high similarity titles
        ids_to_remove = []
        grouped = df_cleaning.groupby(['07_Item', 'Scraped_Price'])

        for name, group in grouped:
            if len(group) > 1:
                similarity = self._calculate_title_similarity(group['Scraped_Name'].dropna().tolist())
                if similarity > 80:
                    ids_to_remove.extend(group.index.tolist()[1:])

        ids_to_remove = list(set(ids_to_remove))
        df_cleaning = df_cleaning.drop(index=ids_to_remove)

        # 3. Apply price filters
        rows_to_keep = pd.Series(True, index=df_cleaning.index)

        for item, filters in self.price_filters.items():
            item_mask = (df_cleaning['06_Group'] == item)
            
            min_p_list = filters.get('min_p', [])
            if min_p_list:
                rows_to_keep[item_mask & (df_cleaning['Scraped_Price'] < min_p_list[0])] = False
                
            max_p_list = filters.get('max_p', [])
            if max_p_list:
                rows_to_keep[item_mask & (df_cleaning['Scraped_Price'] > max_p_list[0])] = False

        df_cleaning = df_cleaning[rows_to_keep]

        # 4. Update validated dataframe
        prices_df['validated'] = 0
        prices_df.loc[prices_df.index.isin(df_cleaning.index), 'validated'] = 1
        
        prices_df.to_excel(self.data_path_prices, index=False)
        print(f"Cleaning completed: {len(df_cleaning)} valid items retained out of {len(prices_df)}.")
        
        return prices_df

# This conditional block ensures you can still test this file standalone if needed,
# but it won't execute when main.py imports it.
if __name__ == "__main__":
    from src.config import PATHS, CODES
    
    # 1. Test the extraction phase
    scraper = EconomicScraper(PATHS, CODES)
    # DEBUG: _prepare_search_data
    # df_search = scraper._prepare_search_data()
    # DEBUG: Prepare search datasets within run
    # df_search, amazon_items_df, wallapop_items_df = scraper.run(extract_amazon=False, extract_wallapop=False)
    # DEBUG: _extract_raw_prices_source_a
    # prices_df = scraper._extract_raw_prices_source_a(amazon_items_df)
    # DEBUG: _extract_raw_prices_source_b
    # prices_df = scraper._extract_raw_prices_source_b(wallapop_items_df, prices_df)
    scraper.run()
    
    # 2. Test the cleaning phase
    cleaner = PriceDataCleaner(PATHS)
    cleaner.clean_data()
    