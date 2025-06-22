from tools import * 
import pandas as pd 
import os 
# os.chdir(os.getcwd() + '/SAE_analysis/')

mass_csv = pd.read_csv(os.getcwd() + '/data/mass_sae_project.csv')
mass_csv = clean_indexes(mass_csv)
mass_csv.columns
mass_csv['dom_tags_dict'] = mass_csv['dom_tags'].apply(lambda x: parse_dom_tags(x))

mass_csv[['secondary_urls', 'init_texts']] = mass_csv[['url_extraction', 'dom_tag_types', 'multi_page', 'multi_page_num', 'dom_tags_dict']].apply(
    lambda row: dynamic_scraper(url=row['url_extraction'], num=int(row['multi_page_num']), dom_header_type=row['dom_tag_types'],
                                multi_page=bool(row['multi_page']), **row['dom_tags_dict']), axis=1
).apply(pd.Series)

mass_csv = mass_csv.dropna(subset=['secondary_urls'])
mass_csv['secondary_urls_rem_dup'] = mass_csv['secondary_urls'].apply(lambda url: dedup_list(url))
mass_csv = mass_csv.dropna(subset=['secondary_urls_rem_dup'])

print("Total number of documents:", mass_csv['valid_secondary_urls'].apply(lambda x: len(x) if x else 0).sum())

