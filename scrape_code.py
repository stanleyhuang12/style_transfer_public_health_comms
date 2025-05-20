from tools import * 
import requests
import pandas as pd 
import os 

# Scrape dataset needed for style transfer tasks 

texas_csv = pd.read_csv(os.getcwd() + '/data/texas_sae_project.csv')
texas_w_csv = pd.read_csv(os.getcwd()+'/data/texas_sae_project_w_links.csv')
texas_w_csv.columns = [col for col in texas_w_csv.columns.str.strip()]

mass_csv = pd.read_csv(os.getcwd() + '/data/mass_sae_project.csv')
mass_csv['State'] = "MA"


# Capture language from STRIPED's campaign 

striped_ookh_website = "https://hsph.harvard.edu/research/eating-disorders-striped/policy-translation/out-of-kids-hands/"
striped_ma_website = "https://hsph.harvard.edu/research/eating-disorders-striped/policy-translation/out-of-kids-hands/massachusetts/"

# Nested website URLs, nested texts 
striped_web_urls, striped_texts = complete_text_url_extraction(striped_ma_website, 'div', class_='wp-block-column is-layout-flow wp-block-column-is-layout-flow', style='flex-basis:66.66%')
_, striped_general_texts = complete_text_url_extraction(striped_ookh_website, 'div', class_='wp-block-column is-layout-flow wp-block-column-is-layout-flow', style='flex-basis:66.66%')

# Construct dataset 
dataset = pd.DataFrame(columns=['url', 'organization_name','texts', 'state', 'year_established', 'topic', 'region_state', 'org_type', 'polit_aff', 'text_modal', 'polarity', 'sentiment'])

## Data Entry requires us to specify topic (if possible), 'text_modality', 'polarity', 'sentiment' if possible 
striped_entry = data_entry(striped_general_texts, mass_csv.iloc[0], text_modal='website')
striped_entry_2 = data_entry(striped_texts, mass_csv.iloc[0], text_modal='website')
dataset = dataset._append(pd.Series(striped_entry), ignore_index=True)
dataset = dataset._append(pd.Series(striped_entry_2), ignore_index=True)




# Take a website
## Extract all relevant texts
## Extract all relevant urls 
## 
rows, left_over_url = iterate_through_urls_with_pdfs(striped_web_urls)

website_extraction = []
rows = []
for url in striped_web_urls: 
    if _is_downloadable(url): 
        extracted_texts = extract_text_from_pdfs(url)
        text_dct = {
            'topic': 'diet_supplements',
            'jurisdiction': 'MA',
            'polit_affil': None,
            'doc_type': 'pdf',
            'genre': 'academic',
            'polarity': None,
            'sentiment': None,
            'original_language': 1,
            'texts': extracted_texts
        }
    
        rows.append(text_dct)
    else: 
        website_extraction.append(url)
        
        

dataset = pd.concat((dataset, pd.DataFrame(rows))).reset_index(drop=True)
dataset['texts'] = dataset['texts'].apply(lambda x: x.strip())

urls, texts  = complete_text_url_extraction('https://malegislature.gov/Bills/194/HD716', 'div', class_="col-xs-12 col-md-8")


