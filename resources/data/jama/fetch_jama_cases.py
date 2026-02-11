import cloudscraper
from bs4 import BeautifulSoup
import json
from datetime import datetime
from tqdm import tqdm 
# URL of the JAMA Network Clinical Challenges page
BASE_URL = "https://jamanetwork.com/collections/44038/clinical-challenge?f_SiteID=16&fl_Categories=Clinical+Challenge&fl_ContentType=Article" # 

# Function to scrape the clinical cases
def scrape_clinical_cases():
    scraper = cloudscraper.create_scraper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    cases_links = []
    page_number = 1
    
    while True:
        url = f"{BASE_URL}&page={page_number}"
        
        response = scraper.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        
        case_elements = soup.find_all("li", class_="article") 
        
        if not case_elements:
            break

        links_found_this_page = 0
        for idx, case in enumerate(case_elements):
            link_tag = case.find("a", class_="article--title") 
            link = link_tag['href'] if link_tag else None

            date_tag = case.find("div", class_="article--date meta-item no-wrap")
            date_text = date_tag.text.strip() if date_tag else None
            
            if date_text:
                try:
                    publication_date = datetime.strptime(date_text, "%B %d, %Y")
                    if publication_date.year < 2021:
                        return cases_links 
                except ValueError:
                    pass
            
            if link:
                cases_links.append(link)
                links_found_this_page += 1

        print(f"Case URLs of page {page_number} fetched...")
        page_number += 1

    return cases_links

# Main execution
if __name__ == "__main__":
    clinical_cases_links = scrape_clinical_cases()
    
    compiled_results = []
    
    for idx, link in tqdm(enumerate(clinical_cases_links), total=len(clinical_cases_links)):
        compiled_results.append({
            'id': idx,
            'link': link
        })
    
    # Save the results to a JSON file
    file_name = 'jama_links_updated.json'
    with open(file_name, 'w') as json_file:
        json.dump(compiled_results, json_file, indent=4)

    print(f"Saved {len(compiled_results)} cases to {file_name}")
