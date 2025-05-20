import requests
from bs4 import BeautifulSoup 
from urllib.parse import urlparse, urljoin
import pymupdf
import pytesseract
import io 
from PIL import Image


def complete_text_url_extraction(url: str, dom_header_type: str, **kwargs): 
    """Takes a URL, html_elements, and additional parameters like style, class, id and scrapes all the nested text 
    and URLs. Returns all the relevant texts and list of URLs. 

    Args:
        url (str): A URL of the website page 
        dom_header_type (str): HTML element type (e.g., div, p, header)

    Returns:
        _type_: _description_
    """
    
    html_response = requests.get(url)
    parser = BeautifulSoup(html_response.content, 'html.parser')
    
    header = parser.find(dom_header_type, **kwargs)
    
    # Extract all urls 
    sourced_links = header.find_all('a', href=True)
    print("Retrieving all URLs...")
    
    retrieved_urls = [link.get('href') for link in sourced_links]
    
    # Extract all texts
    
    print("Extracting all texts...")
    texts = header.get_text(separator='\n', strip=True)
    
    return retrieved_urls, texts


def _is_downloadable(url: str) -> bool:
    """Checks if URL is a downloadable extension. Returns a boolean value type 

    Args:
        url (str): A URL of the website.

    Returns:
        bool: A True if the URL ends with a downloadble extension. 
    """
    download_exts = ('.pdf', '.zip', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                     '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.csv', '.txt')
    
    path = url.lower()
    
    return True if path.endswith(download_exts) else False 
    

def ocr_extract_for_images(url: str): 
    
    valid_ext = ('.img', '.png', '.jpg', '.jpeg')

    if url.endswith(valid_ext): 
        
        response = requests.get(url)
        
        if response.status_code != 200: 
            print(f'Status code error {response.status_code} for {url}')
            
            return ""

        data = response.content 
        streamed_image = Image.open(io.BytesIO(data))
        
        return pytesseract.image_to_string(streamed_image) 
    
    else: 
        
        return " "
        

# The first step is to handle non-HTML website items 

def extract_text_from_pdfs(url: str): 
    """Handles non-HTML extensions and extract all texts from the PDF. 

    Args:
        url (str): A URL of the website.

    Returns:
        _type_: _description_
    """
    ## validation
    valid_ext_set = ('.pdf', '.txt')
    
    if url.endswith(valid_ext_set): 
        response = requests.get(url)
        
        if response.status_code != 200: 
            print(f"Error: Received status code {response.status_code}")
            return ""
        
        stream_data = response.content

        # Stream the file data
        # Resource: https://pymupdf.readthedocs.io/en/latest/how-to-open-a-file.html
        doc = pymupdf.Document(stream=stream_data)
        
        ## Exclude extraordinarily lengthy PDFs 
        if doc.page_count > 20: 
            pass 
        # Resource: https://pymupdf.readthedocs.io/en/latest/app1.html 
        return "\n".join(page.get_text("text", 
                                       flags=pymupdf.TEXT_INHIBIT_SPACES & ~pymupdf.TEXT_PRESERVE_WHITESPACE) 
                         for page in doc)
    else: 
        return ""

def iterate_through_urls_with_pdfs(list_of_urls): 

    not_downloadable = []
    rows = [ ]
    
    for url in list_of_urls: 
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
            not_downloadable.append(url)
            
    return rows, not_downloadable 
    
    



def url_validation(url: str) -> str: 
    
    http_start = ('https://', 'http://')
    if url.startswith(http_start): 
        
        html_response = requests.get(url)
    
        if html_response.status_code in [200, 301]: 
                return url
        else: 
            print('Error retrieving content from URL:', html_response.status_code)
            return None 

    parsed_url = urlparse(url)
    base_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
    joined_url = urljoin(base_url, parsed_url)
    try:
        response = requests.get(joined_url, timeout=5)
        if response.status_code in [200, 301]:
            return joined_url
    except requests.RequestException:
        pass
       
    inherited_with_path = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    joined_with_path = urljoin(inherited_with_path, url)
    try:
        response = requests.get(joined_with_path, timeout=5)
        if response.status_code in [200, 301]:
            return joined_with_path
    except requests.RequestException:
        pass

    print(f"Failed to validate URL: {url}")
    return None

                       

def data_entry(texts, row, **kwargs): 
    """Takes the text and annotates it with the corresponding attributes"""
    
    data_dictionary = {
        "url": row['URL'],
        "sub_url": row['URL_sub'],
        "organization_name": row['Organization'],
        "org_type": row['Type'],
        "polit_aff": row['Political Affiliation'],
        "year_established": row['Year Established'],
        "region_state": row['Region of State'],
        "state": row['State'],
        "texts": texts,
        "topic": None,
        "text_modal": None,
        "polarity": None,
        "sentiment": None   
    }
    
    data_dictionary.update(kwargs)
    
    return data_dictionary

