#!/usr/bin/env python3
"""
HPO Disease Information Scraper
A professional tool to scrape disease information from the Human Phenotype Ontology (HPO) website

Usage:
    python hpo_scraper.py <disease_id>
    
Example:
    python hpo_scraper.py OMIM:176270
    python hpo_scraper.py ORPHA:739
    python hpo_scraper.py MONDO:0008300
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
import sys
import argparse
from urllib.parse import urljoin

# Try to import selenium for JavaScript rendering
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

def scrape_hpo_disease_with_selenium(disease_url):
    """
    Scrape disease overview information using Selenium for JavaScript rendering
    
    Args:
        disease_url (str): The direct URL to the disease page
        
    Returns:
        dict: Dictionary containing disease name and overview information
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Create driver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Navigate to the page
            driver.get(disease_url)
            
            # Wait for the page to load
            wait = WebDriverWait(driver, 20)
            
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            except:
                pass
            
            # Get page source
            page_source = driver.page_source
            disease_soup = BeautifulSoup(page_source, 'html.parser')
            
            disease_info = {
                'disease_name': '',
                'overview': '',
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_url': disease_url,
                'method': 'selenium'
            }
            
            # Extract disease name from h1 tag
            title_element = disease_soup.find('h1')
            if title_element:
                disease_info['disease_name'] = title_element.get_text().strip()
            
            # Extract overview information from mat-list-item/span
            mat_list_items = disease_soup.find_all('mat-list-item')
            
            for item in mat_list_items:
                span_elements = item.find_all('span')
                for span in span_elements:
                    text = span.get_text().strip()
                    if text and len(text) > 10:
                        disease_info['overview'] = text
                        break
                if disease_info['overview']:
                    break
            
            return disease_info
            
        finally:
            driver.quit()
            
    except Exception as e:
        return None

def scrape_hpo_disease_overview_improved(disease_url):
    """
    Improved function to scrape disease overview information with better accuracy
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Create a session to maintain cookies
        session = requests.Session()
        session.headers.update(headers)
        
        # First, try to access the main HPO page to get cookies
        main_page_response = session.get("https://hpo.jax.org/", timeout=30)
        
        # Get the disease page
        disease_response = session.get(disease_url, timeout=30)
        
        if disease_response.status_code == 404:
            # Try alternative URL formats
            alternative_urls = [
                disease_url.replace("https://hpo.jax.org/browse/disease/", "https://hpo.jax.org/app/browse/disease/"),
                disease_url.replace("browse/disease", "browse/disease/"),
                f"https://hpo.jax.org/app/browse/disease/{disease_url.split('/')[-1]}"
            ]
            
            for alt_url in alternative_urls:
                alt_response = session.get(alt_url, timeout=30)
                if alt_response.status_code == 200:
                    disease_response = alt_response
                    disease_url = alt_url
                    break
        
        disease_response.raise_for_status()
        
        disease_soup = BeautifulSoup(disease_response.content, 'html.parser')
        
        disease_info = {
            'disease_name': '',
            'overview': '',
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_url': disease_url,
            'method': 'improved_requests'
        }
        
        # Extract disease name from h1 tag
        title_element = disease_soup.find('h1')
        if title_element:
            disease_info['disease_name'] = title_element.get_text().strip()
        
        # Look for mat-list-item/span elements (the specific location for disease descriptions)
        mat_list_items = disease_soup.find_all('mat-list-item')
        
        for item in mat_list_items:
            # Look for span elements within mat-list-item
            span_elements = item.find_all('span')
            for span in span_elements:
                text = span.get_text().strip()
                # If we find text in a span within mat-list-item, use it
                if text and len(text) > 10:  # Basic length check
                        disease_info['overview'] = text
                        break
                if disease_info['overview']:
                    break
        
        return disease_info
        
    except requests.RequestException as e:
        return None
    except Exception as e:
        return None

def scrape_hpo_disease_with_click_simulation(disease_url):
    """
    Use Selenium to simulate clicks and get HPO disease overview information
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    try:
        # Setup Firefox options
        from selenium.webdriver.firefox.options import Options as FirefoxOptions
        firefox_options = FirefoxOptions()
        firefox_options.add_argument('--headless')
        firefox_options.add_argument('--width=1920')
        firefox_options.add_argument('--height=1080')
        
        # Create driver with custom geckodriver path
        from selenium.webdriver.firefox.service import Service
        service = Service(executable_path=os.path.expanduser("~/geckodriver"))
        driver = webdriver.Firefox(service=service, options=firefox_options)
        
        try:
            # First access HPO main page
            driver.get("https://hpo.jax.org/")
            time.sleep(3)
            
            # Try to search for the disease
            try:
                search_box = driver.find_element(By.XPATH, "//input[@type='text' or @type='search']")
                disease_id = disease_url.split('/')[-1]
                search_box.send_keys(disease_id)
                time.sleep(2)
                
                try:
                    search_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
                    search_button.click()
                except:
                    search_box.send_keys(Keys.RETURN)
                
                time.sleep(5)
                
                # Look for disease links in search results
                disease_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'disease')]")
                if disease_links:
                    disease_links[0].click()
                    time.sleep(5)
                else:
                    driver.get(disease_url)
                    time.sleep(5)
                    
            except Exception as e:
                driver.get(disease_url)
                time.sleep(5)
            
            # Wait for page to load
            wait = WebDriverWait(driver, 20)
            
            # Try to click various possible buttons to expand content
            try:
                expand_selectors = [
                    "//button[contains(text(), 'Show')]",
                    "//button[contains(text(), 'Expand')]", 
                    "//button[contains(text(), 'More')]",
                    "//button[contains(text(), 'View')]",
                    "//button[contains(@class, 'expand')]",
                    "//button[contains(@class, 'show')]",
                    "//div[contains(@class, 'expandable')]",
                    "//span[contains(text(), '...')]"
                ]
                
                for selector in expand_selectors:
                    try:
                        buttons = driver.find_elements(By.XPATH, selector)
                        for button in buttons:
                            try:
                                driver.execute_script("arguments[0].click();", button)
                                time.sleep(2)
                            except:
                                continue
                    except:
                        continue
            except:
                pass
            
            # Try to click tabs
            try:
                tab_keywords = ['overview', 'summary', 'description', 'definition', 'clinical', 'phenotype']
                for keyword in tab_keywords:
                    try:
                        tabs = driver.find_elements(By.XPATH, f"//*[contains(text(), '{keyword}') and (self::button or self::div or self::span)]")
                        for tab in tabs:
                            try:
                                if tab.is_displayed() and tab.is_enabled():
                                    driver.execute_script("arguments[0].click();", tab)
                                    time.sleep(3)
                                    break
                            except:
                                continue
                    except:
                        continue
            except:
                pass
            
            # Get page source
            page_source = driver.page_source
            disease_soup = BeautifulSoup(page_source, 'html.parser')
            
            disease_info = {
                'disease_name': '',
                'overview': '',
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_url': disease_url,
                'method': 'selenium_click_simulation'
            }
            
            # Extract disease name
            title_selectors = ['h1', 'h2', '.title', '.disease-name', '[class*="title"]']
            for selector in title_selectors:
                title_element = disease_soup.select_one(selector)
                if title_element:
                    disease_info['disease_name'] = title_element.get_text().strip()
                    break
            
            # Extract overview information using the specific XPath
            overview = extract_disease_description_from_selenium(driver, disease_soup)
            if overview:
                disease_info['overview'] = overview
            
            return disease_info
            
        finally:
            driver.quit()
            
    except Exception as e:
        return None

def extract_disease_description_from_selenium(driver, soup):
    """Extract disease description information from Selenium page"""
    try:
        # Use the exact XPath path specified by user
        target_xpath = "/html/body/app-root/mat-sidenav-container/mat-sidenav-content/div/app-disease/div/div/div[2]/div/div/mat-card/mat-card-content/mat-list/mat-list-item/span"
        
        try:
            # Try to use Selenium to directly locate the specific span element
            target_element = driver.find_element(By.XPATH, target_xpath)
            if target_element:
                text = target_element.text.strip()
                # If we get text from the exact location, return it (even if short)
                # This means the website has content at this location
                if text:
                    return text
        except Exception as e:
            # If we can't find the exact element, try the mat-list-item level
            try:
                fallback_xpath = "/html/body/app-root/mat-sidenav-container/mat-sidenav-content/div/app-disease/div/div/div[2]/div/div/mat-card/mat-card-content/mat-list/mat-list-item"
                target_element = driver.find_element(By.XPATH, fallback_xpath)
                if target_element:
                    text = target_element.text.strip()
                    if text:
                        return text
            except Exception as e2:
                pass
        
        # If we can't find content at the specified location, return None
        # This means the disease has no description on the website
        return None
        
    except Exception as e:
        return None

def scrape_orphanet_disease_description(disease_id):
    """
    Scrape disease description from Orphanet website for ORPHA disease IDs
    
    Args:
        disease_id (str): ORPHA disease ID (e.g., ORPHA:398069)
        
    Returns:
        dict: Dictionary containing disease information from Orphanet
    """
    try:
        if not disease_id.startswith('ORPHA:'):
            return None
            
        # Extract the ORPHA number
        orpha_number = disease_id.split(':')[1]
        
        # Build Orphanet URL
        orphanet_url = f"https://www.orpha.net/en/disease/detail/{orpha_number}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Create a session to maintain cookies
        session = requests.Session()
        session.headers.update(headers)
        
        # Get the Orphanet disease page
        response = session.get(orphanet_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        disease_info = {
            'disease_name': '',
            'overview': '',
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_url': orphanet_url,
            'method': 'orphanet_scraping'
        }
        
        # Extract disease name from the title
        title_element = soup.find('h1')
        if title_element:
            disease_info['disease_name'] = title_element.get_text().strip()
        
        # Try to extract description using the specified XPath logic
        # Parent node: /html/body/main/article/article/div/main/div/div/div/div/div[4]
        # Keyword node: /html/body/main/article/article/div/main/div/div/div/div/div[4]/strong
        # Description node: /html/body/main/article/article/div/main/div/div/div/div/div[4]/p
        try:
            # Navigate through the structure step by step to find div[4] (parent node)
            # /html/body/main/article/article/div/main/div/div/div/div/div[4]
            body = soup.find('body')
            if body:
                main1 = body.find('main')
                if main1:
                    articles = main1.find_all('article')
                    if len(articles) >= 2:
                        article2 = articles[1]  # Second article
                        divs = article2.find_all('div')
                        
                        # Find div that contains main
                        for div in divs:
                            inner_main = div.find('main')
                            if inner_main:
                                inner_divs = inner_main.find_all('div')
                                if len(inner_divs) >= 5:  # Need at least 5 divs to get div[4]
                                    # Check all divs from 0 to 4 to find the one with "Disease definition"
                                    for i in range(min(5, len(inner_divs))):
                                        target_div = inner_divs[i]
                                        
                                        # Check keyword node: /div[i]/strong contains "Disease definition"
                                        strong_element = target_div.find('strong')
                                        if strong_element:
                                            strong_text = strong_element.get_text().strip()
                                            if 'Disease definition' in strong_text:
                                                # Found "Disease definition", now extract from description node: /div[i]/p
                                                paragraphs = target_div.find_all('p')
                                                if paragraphs:
                                                    # Get the first paragraph from description node
                                                    description_text = paragraphs[0].get_text().strip()
                                                    if description_text:
                                                        disease_info['overview'] = description_text
                                                        break
                                    break
        except Exception as e:
            pass
        
        return disease_info
        
    except requests.RequestException as e:
        return None
    except Exception as e:
        return None

def filter_out_identifiers(text):
    """Filter out identifiers and clean up text"""
    try:
        import re
        
        # Remove various identifier formats
        text = re.sub(r'MONDO:\d+\s*open_in_new\s*', '', text)
        text = re.sub(r'OMIM:\d+\s*open_in_new\s*', '', text)
        text = re.sub(r'ORPHA:\d+\s*open_in_new\s*', '', text)
        text = re.sub(r'HPO:\d+\s*open_in_new\s*', '', text)
        
        # Remove common webpage elements
        webpage_elements = [
            'get_app', 'Export', 'Associations', 'bug_report', 'Report Entry Issue',
            'open_in_new', 'more_horiz', 'expand_more', 'chevron_right',
            'search', 'filter_list', 'settings', 'info', 'help'
        ]
        
        for element in webpage_elements:
            text = re.sub(rf'\b{element}\b', '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Check if the text is meaningful (not just webpage elements)
        if len(text) < 30:  # Too short to be a meaningful description
            return None

        # Check if it contains mostly webpage elements
        meaningful_words = len([word for word in text.split() if len(word) > 3])
        if meaningful_words < 3:  # Not enough meaningful words
            return None

        return text
            
    except Exception as e:
        return text

def get_database_source_from_url(source_url):
    """
    Get database source information based on the actual source URL where description was extracted
    
    Args:
        source_url (str): The URL where the description was successfully extracted
        
    Returns:
        str: Simple database name
    """
    if 'orpha.net' in source_url:
        return 'ORPHA'
    elif 'hpo.jax.org' in source_url:
        return 'HPO'
    else:
        return 'UNKNOWN'

def get_disease_description(disease_id):
    """
    Get disease description based on disease ID
    
    Args:
        disease_id (str): Disease ID, supports formats like:
            - OMIM:176270
            - ORPHA:739
            - MONDO:0008300
    
    Returns:
        dict: Dictionary containing disease name and description with status information
        - If successful with content: returns dict with 'description' field
        - If successful but no content: returns dict with 'description': '' and 'status': 'no_content'
        - If failed: returns dict with 'status': 'failed' and 'error': error_message
    """
    try:
        # For ORPHA disease IDs, try Orphanet first
        if disease_id.startswith('ORPHA:'):
            orphanet_result = scrape_orphanet_disease_description(disease_id)
            
            if orphanet_result:
                disease_name = orphanet_result.get('disease_name', '')
                description = orphanet_result.get('overview', '')
                
                if description:  # Successfully extracted content from Orphanet
                    source_url = orphanet_result.get('source_url', '')
                    return {
                        'disease_id': disease_id,
                        'disease_name': disease_name,
                        'description': description,
                        'source_url': source_url,
                        'scraped_at': orphanet_result.get('scraped_at', ''),
                        'source_database': get_database_source_from_url(source_url),
                        'status': 'success'
                    }
                # If Orphanet returns empty description, continue to try HPO as fallback
        
        # Build HPO URL for all disease types (including ORPHA as fallback)
        if disease_id.startswith('OMIM:'):
            url = f"https://hpo.jax.org/app/browse/disease/{disease_id}"
        elif disease_id.startswith('ORPHA:'):
            url = f"https://hpo.jax.org/app/browse/disease/{disease_id}"
        elif disease_id.startswith('MONDO:'):
            url = f"https://hpo.jax.org/app/browse/disease/{disease_id}"
        else:
            return {
                'disease_id': disease_id,
                'status': 'failed',
                'error': f'Unsupported disease ID format: {disease_id}'
            }
        
        # Try Selenium scraping first
        result = scrape_hpo_disease_with_click_simulation(url)
        
        if result:
            disease_name = result.get('disease_name', '')
            description = result.get('overview', '')
            
            if description:  # Successfully extracted content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': description,
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'success'
                }
            else:  # Successfully accessed page but no content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': '',
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'no_content'
                }
        
        # Fallback to improved requests method
        result = scrape_hpo_disease_overview_improved(url)
        
        if result:
            disease_name = result.get('disease_name', '')
            description = result.get('overview', '')
            
            if description:  # Successfully extracted content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': description,
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'success'
                }
            else:  # Successfully accessed page but no content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': '',
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'no_content'
                }
        
        # Final fallback to basic Selenium method
        result = scrape_hpo_disease_with_selenium(url)
        
        if result:
            disease_name = result.get('disease_name', '')
            description = result.get('overview', '')
            
            if description:  # Successfully extracted content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': description,
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'success'
                }
            else:  # Successfully accessed page but no content
                return {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'description': '',
                    'source_url': url,
                    'scraped_at': result.get('scraped_at', ''),
                    'source_database': get_database_source_from_url(url),
                    'status': 'no_content'
                }
        
        # All methods failed to access the page
        return {
            'disease_id': disease_id,
            'status': 'failed',
            'error': 'All scraping methods failed to access the page'
        }
            
    except Exception as e:
        return {
            'disease_id': disease_id,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='HPO Disease Information Scraper - Get disease descriptions from HPO website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hpo_scraper.py OMIM:176270
  python hpo_scraper.py ORPHA:739
  python hpo_scraper.py MONDO:0008300
        """
    )
    
    parser.add_argument('disease_id', 
                       help='Disease ID (e.g., OMIM:176270, ORPHA:739, MONDO:0008300)')
    
    parser.add_argument('--output', '-o',
                       help='Output file path (JSON format)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Fetching disease description for: {args.disease_id}")
    
    # Get disease description
    result = get_disease_description(args.disease_id)
    
    if result['status'] == 'success':
        if args.output:
            # Save to file
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to: {args.output}")
        else:
            # Print to stdout
            if args.verbose:
                print(f"Disease ID: {result['disease_id']}")
                print(f"Disease Name: {result['disease_name']}")
                print(f"Source URL: {result['source_url']}")
                print(f"Scraped At: {result['scraped_at']}")
                print(f"Status: {result['status']}")
                print(f"Description: {result['description']}")
            else:
                print(result['description'])
    elif result['status'] == 'no_content':
        if args.verbose:
            print(f"Disease ID: {result['disease_id']}")
            print(f"Disease Name: {result['disease_name']}")
            print(f"Source URL: {result['source_url']}")
            print(f"Scraped At: {result['scraped_at']}")
            print(f"Status: {result['status']}")
            print("Description: (No description available on HPO website)")
        else:
            print("(No description available)")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to: {args.output}")
    else:  # status == 'failed'
        print(f"Failed to get disease description for {args.disease_id}")
        print(f"Error: {result['error']}")
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Error details saved to: {args.output}")
        sys.exit(1)

if __name__ == "__main__":
    main()