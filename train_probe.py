
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model
from tools import * 
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
import numpy as np
import json
import os
from dotenv import load_dotenv, find_dotenv
from baukit import Trace, TraceDict # figure out how to use this 
import warnings
# import spaces

# warnings.filterwarnings('ignore')

llama_model_name = "meta-llama/Llama-3.1-8B-Instruct" 
env_path = find_dotenv()
if load_dotenv(env_path): 
    HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=HF_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, 
                                                   token=HF_TOKEN, 
                                                   torch_dtype= torch.bfloat16,
                                                   ).to(device) 
# Q: What does disk offloading mean? 

gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
gpt_model = GPT2Model.from_pretrained("openai-community/gpt2")

for mod in llama_model.named_modules(): 
    print(mod[1])
## How do i use tracedict to hook onto a mdoule? 

inputs = llama_tokenizer('This is a sentence I am passing into an AI model!', return_tensors='pt')
llama_model.eval()

with Trace(llama_model, 'model.layers.0.self_attn.o_proj', retain_output=True) as tracer:
    _ = llama_model(inputs)
    attn_out = tracer['model.layers.0.self_attn.o_proj']
    


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)

"""
GPT2Model(
  (wte): Embedding(50257, 768) ###NOTE Has a dictionary of 50K words -> ####NOTE:retrieves corresponding trained embedding vectors
  (wpe): Embedding(1024, 768) ###NOTE Appends the word's corresponding position to the corresponding words 
  (drop): Dropout(p=0.1, inplace=False) ####NOTE Not used when model is set to eval
  (h): ModuleList( ##MODULES
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True) #####NOTE Normalizes each vector
      (attn): GPT2Attention( ##NOTE Multi-head attention 
        (c_attn): Conv1D(nf=2304, nx=768) #####NOTE QKV 
        (c_proj): Conv1D(nf=768, nx=768)  #####NOTE context-aware projections 
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False) 
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True) 
      (mlp): GPT2MLP(
        (c_fc): Conv1D(nf=3072, nx=768)
        (c_proj): Conv1D(nf=768, nx=3072)
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
"""

from sklearn.linear_model import LogisticRegression
from typing import Any
from baukit import Trace 


def load_data(eval_mode=True, shuffle=True): 
    """
    Loads data
    eval_mode: loads only essential columns for analysis 
    shuffle: if dataset will undergo shuffling 
    """

    df = pd.read_csv("data/dataset_with_frames_annot_prep.pkl")
    if eval_mode: 
        frame_cols = [
                "economic", "capacity_and_resources", "morality", 
                "fairness_and_equality", "legality_cons_juris", 
                "policy_prescription_eval","crimes_and_punishment", 
                "security_and_defense", "quality_of_life", "political",
                "cultural_identity", "public_opinion", "health_and_safety",
                "external_regulation", "other_frames", "none_detected"
        ]
        frame_col_bins = [f"{col}_bin" for col in frame_cols]
        df = df[frame_col_bins + ['all_frames','organization', 'state', 'political_affiliation']]
        df = df.rename(
            columns={
            'all_frames': 'message', 
            'politcal_affiliation': 'party',
            'state': 'state', 
            'organization': 'org'
        })       
    return df.sample(frac=1, random_state=13).reset_index() if shuffle else df

advocacy_corpus = load_data()
region_map = {
    "MA": 0,
    "TX": 1
}

def convert_dataframe_to_json(df): 
    d = df.to_json(orient='records')
    return json.loads(d)

advocacy_dict = convert_dataframe_to_json(advocacy_corpus)
advocacy_dict[0]

from transformers import pipelines
pipelines("named")
def extract_hidden_reprs(message_dict,
                         tokenizer, 
                         model):     
    representations = {}
    
    for m in tqdm(message_dict): 
        representations[m['index']] = {}
        
        inputs = tokenizer(m['message'], 
                           return_tensors='pt', 
                           truncation=True)  # Move tensor to CUDA if possible? 
        
        with torch.no_grad(): 
            out = model(**inputs, 
                  output_hidden_states=True, 
                  ) # Move model to cuda if possible 
            
            for i, o in enumerate(out.hidden_states[1:]): # Omit the embedding layer
                representations[
                    m['index']][f'repr_{i}'] = o[-1, -1, :].detach().cpu().clone().to(torch.float) # Extract last token of the sentence
    
    return representations 


representation_dict = extract_hidden_reprs(message_dict=advocacy_dict,
                                           tokenizer=gpt_tokenizer,
                                           model=gpt_model)
ground_truth_dict = {l['index']: l['state'] for l in advocacy_dict}


advocacy_dict.keys()

def split_reprs_to_dfs(repr_dict):
    layer_names = next(iter(repr_dict.values())).keys()  # get repr_1, repr_2, ...
    
    layer_dfs = {}
    for layer in layer_names:
        # Collect vectors for this layer from all inputs
        vectors = [repr_dict[i][layer].numpy() for i in repr_dict]
        df = pd.DataFrame(vectors)
        layer_dfs[layer] = df  # You can access as layer_dfs["repr_1"], etc.

    return layer_dfs

df_dict = split_reprs_to_dfs(representation_dict)
df_dict['repr_1']
advocacy_dict
def train_probe(layer_dfs, labels):
    """
    repr_dict: A dictionary of hidden representations. {index: {"repr_1": [tensor()], "repr_2": [tensor()]}}
    n_layers: An integer of the number of layers to probe
    labels: A dictionary of ground-truth labels or outcome variable {index: 1, index_2: 0, ...}

    Given a dictionary of hidden representations, we will concatenate all the hidden representations of each layer
    to create N (# of layers) datasets. 
    """
    
    
    for layer_name, dataset in tqdm(layer_dfs.items()): 
        assert dataset.index == iter(labels.keys()) # Make sure the keys are in the same order 
        clf = LogisticRegression()
        X = dataset.values
        
        
        labels 
    
    for layers in tqdm(len(n_layers)):  
        df = pd.DataFrame()
        for i, t in repr_dict.items(): 
            = t[f'repr{layers}'].numpy()
            

    
for m in advocacy_dict: 
    print(m['message'])
representations = {}
a_dct = {'index': 5}
representations[a_dct['index']] = {}



