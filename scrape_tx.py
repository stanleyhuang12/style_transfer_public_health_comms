import pandas as pd 
import os 
import time
# os.chdir(os.getcwd() + '/SAE_analysis/')
from tools import * 

texas_csv = pd.read_csv(os.getcwd() + '/data/texas_sae_project.csv')
# texas_csv = texas_csv.drop(['unnamed:_0'], axis=1)
texas_csv = clean_indexes(texas_csv)



"""======================================================================================
=======================WEB SCRAPING ARTICLES/PDF/DOCS/IMG================================
========================================================================================="""


# Construct DOM tags for scraping 
texas_csv['dom_tags_dict'] = texas_csv['dom_tags'].apply(lambda x: parse_dom_tags(x))

# Perform initial web scrape gets the texts and secondary URLs (typically PDFs or redirects to more resources)
texas_csv[['secondary_urls', 'init_texts']] = texas_csv[['url_extraction', 'dom_tag_types', 'multi_page', 'multi_page_num', 'dom_tags_dict']].apply(
    lambda row: dynamic_scraper(url=row['url_extraction'], num=int(row['multi_page_num']), dom_header_type=row['dom_tag_types'],
                                multi_page=bool(row['multi_page']), **row['dom_tags_dict']), axis=1
).apply(pd.Series)

texas_csv = texas_csv.dropna(subset=['secondary_urls'])
texas_csv['secondary_urls_rem_dup'] = texas_csv['secondary_urls'].apply(lambda url: dedup_list(url))
texas_csv = texas_csv.dropna(subset=['secondary_urls_rem_dup'])


# Validate secondary URLs 
texas_csv['valid_secondary_urls'] = texas_csv.apply(
    lambda x: url_validation(x['secondary_urls_rem_dup'], x['url_extraction']), 
    axis=1
)


print("Total number of documents:", texas_csv['valid_secondary_urls'].apply(lambda x: len(x) if x else 0).sum())


cols_w_annotations= ['organization', 'url', 'url_extraction', 'type', 'political_affiliation', 'year_established', 'region_of_state', 'state', 'valid_secondary_urls', 'dom_tags_dict', 'dom_tag_types']
annot_df = texas_csv[cols_w_annotations].explode('valid_secondary_urls').reset_index(drop=True)

# For each validated URL, we will now scrape any files or PDF texts
annot_df = annot_df[(annot_df['valid_secondary_urls'].notnull())]
annot_df['extracted_secondary_texts'] = annot_df['valid_secondary_urls'].apply(lambda x: handle_downloadable_websites(x))


annot_df['extracted_secondary_texts'].isnull().sum()
"""===================================================================================
=======================HANDLE DOWNSTREAM HTML WEBSITES================================
======================================================================================"""

## The previous pipeline extracts texts from PDFs, Word Documents. There are a few edge cases of secondary urls that point
## to HTML websites, we will scrape each of them individually to form our complete corpus dataset 

missing_counts_df = (
    annot_df[['url', 'url_extraction', 'extracted_secondary_texts']]
    .groupby('url_extraction')['extracted_secondary_texts']
    .apply(lambda x: x.isnull().sum())
    .reset_index(name='missing_count')
)

print(missing_counts_df.sort_values('missing_count', ascending=False).reset_index(drop=True))
missing_counts_df = missing_counts_df.sort_values('missing_count', ascending=False).reset_index(drop=True)
missing_counts_df['dom_tags_type'] = 'div'


et_org_blog = "https://everytexan.org/our-blog/1/ "
et_org_test = "https://everytexan.org/testimony/1/ "
th_htc_page = "https://taylorhooton.org/hootscorner/page/1/"
tm_org_tob = "https://www.texmed.org/Search/Keywords/?keyword=Tobacco"
tm_org_ecig = "https://www.texmed.org/Search/Keywords/?keyword=E-Cigarettes"

et_blog_idx = annot_df[(annot_df['extracted_secondary_texts'].isnull()) & (annot_df['url_extraction'] == et_org_blog)].index
et_test_idx = annot_df[(annot_df['extracted_secondary_texts'].isnull()) & (annot_df['url_extraction'] == et_org_test)].index
th_htc_idx = annot_df[(annot_df['extracted_secondary_texts'].isnull()) & (annot_df['url_extraction'] == th_htc_page)].index
tm_tob_idx = annot_df[annot_df['extracted_secondary_texts'].isnull() & (annot_df['url_extraction'] == tm_org_tob)].index
tm_cig_idx = annot_df[annot_df['extracted_secondary_texts'].isnull() & (annot_df['url_extraction'] == tm_org_ecig)].index
annot_df.loc[th_htc_idx, 'dom_tag_types'] = 'div'


driver_init = webdriver.Chrome()
annot_df['tert_urls'] = None


results_bl = annot_df.loc[
    et_blog_idx
    ].apply(
        lambda row: complete_text_url_extraction(
            row['valid_secondary_urls'], row['dom_tag_types'], driver=driver_init, class_="elementor-widget-wrap elementor-element-populated"), axis=1
        )
 
annot_df.loc[et_blog_idx,"tert_urls"] = results_bl.apply(lambda x: x[0] if x else None)
annot_df.loc[et_blog_idx, "extracted_secondary_texts"] = results_bl.apply(lambda x: x[1] if x else None)


# Error extracting texts from this group
results_et = annot_df.loc[
    et_test_idx
    ].apply(
        lambda row: complete_text_url_extraction(
            row['valid_secondary_urls'], 'div', driver=driver_init, class_='elementor-widget-text-editor'),
        axis=1
    )

annot_df.loc[et_test_idx, 'extracted_secondary_texts'] = results_et.apply(
    lambda x: x[1] if len(x[0]) == 0 else handle_downloadable_websites(x[0][0])
)


# annot_df.loc[et_test_idx,"tert_urls"] = results_et.apply(lambda x: x[0] if x else None)
# annot_df.loc[et_test_idx, "extracted_secondary_texts"] = results_et.apply(lambda x: x[1] if x else None)
   
# # They have PDFs attached 
# for link, text in results_et: 
#     url = dedup_list(link)
#     text = handle_downloadable_websites(url)
    


results_tm = annot_df.loc[
    tm_tob_idx
    ].apply(
        lambda row: complete_text_url_extraction(
            row['valid_secondary_urls'], row['dom_tag_types'], driver=driver_init, class_="box entry"),
        axis=1)
    
annot_df.loc[tm_tob_idx,"tert_urls"] = results_tm.apply(lambda x: x[0] if x else None)
annot_df.loc[tm_tob_idx, "extracted_secondary_texts"] = results_tm.apply(lambda x: x[1] if x else None)

results_cig = annot_df.loc[tm_cig_idx].apply(
    lambda row: complete_text_url_extraction(
        row['valid_secondary_urls'],
        row['dom_tag_types'],
        driver=driver_init,
        class_="box entry"
    ), axis=1
)

annot_df.loc[tm_cig_idx, "tert_urls"] = results_cig.apply(lambda x: x[0] if x else None)
annot_df.loc[tm_cig_idx, "extracted_secondary_texts"] = results_cig.apply(lambda x: x[1] if x else None)

""" Secondary URLs that had custom HTML content to parse                      # How to extract
0                 https://everytexan.org/our-blog/1/             204    div, class="elementor-widget-container"
1                https://everytexan.org/testimony/1/             192    div, class="elementor-widget-container"
2        https://taylorhooton.org/hootscorner/page/1/            123    div, class="c-blog-single--content"
3   https://www.texasappleseed.org/research-report...             50    pass 
4   https://www.texmed.org/Search/Keywords/?Keywor...             25    div, class="box entry" 
5   https://www.texmed.org/Search/Keywords/?keywor...             16    div, class="box entry" 
6                   https://www.txphc.org/testimonies             16    pass
7             https://texashealthinstitute.org/news/              10    pass 
8                    https://www.txphc.org/resources               6    pass 
9   https://everytexan.org/our-work/policy-areas/h...              5    pass 
10                https://taylorhooton.org/downloads/              4    pass 
11  https://www.texmed.org/Search/Keywords/?keywor...              4    pass 
12  https://www.texmed.org/88thLegislatureTestimon...              3    pass 
13              https://hogg.utexas.edu/toolkits-pubs              2    pass 
14                     https://www.trha.org/advocacy/              1    pass 
15  https://www.texasappleseed.org/research-report...              0    pass 
"""


annot_df.loc[:,'extracted_secondary_texts'].notnull().sum()
annot_df[annot_df['extracted_secondary_texts'].isnull()].loc[19, 'url_extraction']
annot_df.columns

# Save dataset 
annot_df.to_csv(os.getcwd() + '/data/scraped_texts_texas.csv', index=False)
