import pandas as pd
import json 
from tqdm.auto import tqdm 
import torch 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np 
import csv 
import os 
from collections import OrderedDict
from baukit import TraceDict 
import tempfile


def load_data(N, eval_mode=True, shuffle=True): 
    """
    Loads data
    eval_mode: loads only essential columns for analysis 
    shuffle: if dataset will undergo shuffling 
    """
    df = pd.read_pickle("dataset_with_frames_annot_prep.pkl")
    if eval_mode: 
        frame_cols = [
                "economic", "capacity_and_resources", "morality", 
                "fairness_and_equality", "legality_cons_juris", 
                "policy_prescription_eval","crimes_and_punishment", 
                "security_and_defense", "quality_of_life", "political",
                "cultural_identity", "public_opinion", "health_and_safety",
                "external_regulation", "other_frames", "none_detected"]
        frame_col_bins = [f"{col}_bin" for col in frame_cols]
        df = df[frame_col_bins + ['all_frames_preprocessed','organization', 'state', 'political_affiliation']]
        df = df.rename(
            columns={
            'all_frames_preprocessed': 'message', 
            'politcal_affiliation': 'party',
            'state': 'state', 
            'organization': 'org'
        })     
    return df.sample(n=N, random_state=13).reset_index() if shuffle else df


def convert_dataframe_to_json(df): 
    d = df.to_json(orient='records')
    return json.loads(d)

def atomic_pickle_save(obj, filename):
    """Safely write pickle to file, avoiding corruption on interrupt."""
    with tempfile.NamedTemporaryFile('wb', delete=False) as tmp:
        pickle.dump(obj, tmp)
        temp_name = tmp.name
    os.replace(temp_name, filename)
    
def extract_hidden_reprs(message_dict, 
                         tokenizer, 
                         model, 
                         batch_size=16,
                         save_every=100, 
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         checkpoint_file='repr_checkpoint.pkl'): 
    
    model.eval()
    representations = {}

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_file):
        print(f"Resuming from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            representations = pickle.load(f)
        start_index = len(representations)
    else:
        start_index = 0

    message_dict = message_dict[start_index:]

    try:
        for i in tqdm(range(0, len(message_dict), batch_size)):
            batch = message_dict[i:i + batch_size]
            texts = [m['message'] for m in batch]
            indices = [m['index'] for m in batch]

            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

                for b_idx, idx in enumerate(indices):
                    representations[idx] = {}
                    for l_idx, layer_output in enumerate(out.hidden_states[1:]):
                        seq_len = int(inputs['attention_mask'][b_idx].sum().item())
                        last_token_repr = layer_output[b_idx, seq_len - 1, :]
                        representations[idx][f'repr_{l_idx}'] = last_token_repr.cpu().clone()

            # Save checkpoint every N examples
            if (start_index + i + batch_size) % save_every == 0:
                atomic_pickle_save(representations, checkpoint_file)
                print(f"Checkpoint saved at {start_index + i + batch_size} examples")

    except KeyboardInterrupt:
        print("Pause extraction and save current progress")
        atomic_pickle_save(representations, checkpoint_file)
        print("Checkpoint saved before exiting.")

    # Final save
    atomic_pickle_save(representations, checkpoint_file)
    print(f"Final checkpoint saved to {checkpoint_file}")

    return representations
        

# def extract_hidden_reprs(message_dict,
#                          tokenizer, 
#                          model):     
#     representations = {}
    
#     for m in tqdm(message_dict): 
#         representations[m['index']] = {}
        
#         inputs = tokenizer(m['message'], 
#                            return_tensors='pt', 
#                            truncation=True)  # Move tensor to CUDA if possible? 
        
#         with torch.no_grad(): 
#             out = model(**inputs, 
#                   output_hidden_states=True, 
#                   ) # Move model to cuda if possible 
            
#             for i, o in enumerate(out.hidden_states[1:]): # Omit the embedding layer
#                 representations[
#                     m['index']][f'repr_{i+1}'] = o[-1, -1, :].detach().cpu().clone().to(torch.float) # Extract last token of the sentence
    
#     return representations 

def split_reprs_to_dfs(repr_dict):
    layer_names = next(iter(repr_dict.values())).keys()  # get repr_1, repr_2, ...
    
    layer_dfs = {}
    for layer in tqdm(layer_names):
        # Collect vectors for this layer from all inputs
        vectors = [repr_dict[i][layer].to(torch.float16).cpu().numpy() for i in repr_dict]
        df = pd.DataFrame(vectors)
        layer_dfs[layer] = df  # You can access as layer_dfs["repr_1"], etc.

    return layer_dfs


def train_probe(layer_dfs, labels, out_file, os_dir, cv, save_probe=True, random_seed=17):
    """
    layer_dfs: A dict of {layer_name: pd.DataFrame} containing hidden representations
    labels: A dict {index: label} of ground-truth labels
    out_file: Base name for saving probe results and scores
    save_probe: If True, saves the trained logistic regression probe
    random_seed: For reproducibility
    """
    probe_scores = {}
    os.makedirs(os_dir, exist_ok=True)
    out_file = os.path.join(os_dir, out_file)

    y = pd.Series(labels)
 

    for layer_name, dataset in tqdm(layer_dfs.items()):
        # Standardize features before training
        X = dataset.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(random_state=random_seed, max_iter=2000)
        scores = cross_val_score(clf, X_scaled, y, cv=cv)

        # Refit on full data if saving the probe
        if save_probe:
            clf.fit(X_scaled, y)
            with open(f"{out_file}_{layer_name}.pkl", "wb") as o:
                pickle.dump(clf, o)

        # Store scores
        probe_scores[layer_name] = {
            'cv_scores': scores.tolist(),
            'performance': float(np.mean(scores))
        }
        print(f"{layer_name} - CV scores: {scores}, Mean: {np.mean(scores):.4f}")

    # Save scores as CSV
    with open(f"{out_file}_scores.csv", "w", newline='') as res:
        writer = csv.writer(res)
        writer.writerow(["Layer", "CV Scores", "Mean Performance"])
        for layer_name, score_dict in probe_scores.items():
            writer.writerow([layer_name, score_dict['cv_scores'], score_dict['performance']])

def evaluate_probe_classifiers(K, file_dir): 
    """
    Retrieves the top-K best classifier performance and returns the corresponding file paths of the probe models 
    and the names of the layers 
    ## We can improve this code by making name of layers match the name of the layers of the model 
    K: the top-K activation outputs that performs better 
    file_dir: retrieves the file path and loads the CSV score file and appropriate 
    """
    file_paths = os.listdir(file_dir)
    csv_result = [f for f in file_paths if f.endswith(".csv")]
    csv_result_file = os.path.join(file_dir, csv_result[0])
    
    with open(csv_result_file, "r") as infile: 
        res = pd.read_csv(infile)
        sorted_res = res.sort_values("Mean Performance", axis=0, ascending=False)
        print(sorted_res[['Layer', 'Mean Performance']])
        
    top_k_attention_heads = list(sorted_res['Layer'][0:K].values)
    
    model_pathways = []
    
    for attn in top_k_attention_heads: 
        for f in file_paths: 
            if attn in f: 
                model_pathways.append(f)
    
    return list(OrderedDict.fromkeys(model_pathways)), top_k_attention_heads
                

def retrieve_model_coefficients(model_names, file_dir):
    COEFFS_LIST = []
    for path in tqdm(model_names): 
        FILEPATH = os.path.join(file_dir, path)
        
        with open(FILEPATH, "rb") as f: 
            clf = pickle.load(f)
        COEFFS_LIST.append(clf.coef_)
        
    return COEFFS_LIST 



def build_steer_vector(probe_coeffs, alpha=2, norm=False, k=256):
    """
    We build steering vector following the Probe Weight Directions 
    alpha: strength of intervening vector (a scalar)
    torch.register_forward_hook)
    probe_coeffs: a vector of probe coefficients 
    norm: boolean to normalize coeffs if necessary 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if isinstance(probe_coeffs, np.ndarray): 
        print("Converting numpy array coefficients to torch tensors ")
        directions = torch.from_numpy(probe_coeffs).float().to(device)
        
    if isinstance(probe_coeffs, torch.Tensor): 
        directions = probe_coeffs.reshape(-1, 1).to(device)
    
    if norm: 
        directions = directions / directions.norm()
    
    if isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        
    steer_vec = alpha * directions
    _, idx = torch.topk(steer_vec, k=256)
    mask = torch.zeros_like(steer_vec)
    mask[idx] = 1
    
    return steer_vec * mask


def construct_multi_steer_vector(file_dir, top_k, alpha, norm, module=None): 
    """We already constructed N (i.e., # of layers, 32 MHA in Llama 3 7B instruct) probe datasets and run a probe model, 
    store models in a file_dir. This function takes in the directory of the probe models and the results, the # of top k heads
    that you want to intervene on to construct steer models 

    Args:
        best_models (_type_): _description_
        file_dir (_type_): _description_
    """
    best_model_files, layer_names = evaluate_probe_classifiers(K=top_k, file_dir=file_dir) # Return best probe models and respecitve best layers 
    ## Name of best layers are like ['repr_11', 'repr_10', etc...]
    
    coeffs_list = retrieve_model_coefficients(best_model_files, file_dir) # A list of np array of probe coeffs 
    
    steer_vec_dict = {}
    
    if module: 
        layer_vals = [l.strip('repr_') for l in layer_names]
        layer_names = [f"model.layers.{val}.self_attn.o_proj" for val in layer_vals]
    if not module: 
        print("Remember to pass in module to get actual layer names.")
        
    for layer, coeff in zip(layer_names, coeffs_list): 
        steer_vec = build_steer_vector(coeff, alpha=alpha, norm=norm)
        steer_vec_dict[layer] = steer_vec
    
    return steer_vec_dict, layer_names 


def intervene_activations(steer_vec_dict):
    global steer
    steer = steer_vec_dict
    def hook(output, layer_name): 
        steer_vec = steer.get(layer_name, None)
        if steer_vec is not None: 
             steer_vec = steer_vec.to(output.device, dtype=output.dtype)
             output[:, -1, :] = output[:, -1, :] + steer_vec.squeeze(1)
        else: 
            print(f"{layer_name} is not in dictionary")
        return output 
    return hook

    
def generate_messages(module, tokenizer, input_prompt, device, **kwargs): 
    """
    module: model, tokenizer: tokenizer
    input_prompt: input message
    device
    **kwargs: temperature, max_new_tokens, do_sample
    """
    inputs = tokenizer(input_prompt, return_tensors='pt').to(device)
    inputs.input_ids = inputs.input_ids.to(device)
    with torch.no_grad(): 
        model_response = module.generate(**inputs, **kwargs)
    
    response = tokenizer.decode(model_response[0], skip_special_tokens=True)
    print(response)
    return response

global device
device = 'cuda' if torch.cuda.is_available() else "cpu"

def intervene_top_k_attention(module, file_dir, top_k, alpha, tokenizer, input_prompt, device=device, norm=True, **kwargs): 
    
    print("Constructing steer vector from probe coefficients...")
    steer_dict, layers = construct_multi_steer_vector(file_dir=file_dir, module=module, top_k=top_k, alpha=alpha, norm=norm)
   
    hook = intervene_activations(steer_dict)
    
    print("Generating message")
    with torch.no_grad():
        with TraceDict(module=module, layers=layers, edit_output=hook):
           response = generate_messages(module=module, 
                               tokenizer=tokenizer, 
                               input_prompt=input_prompt, 
                               device=device,
                               **kwargs)
    if device: 
        torch.cuda.empty_cache() # release memory from GPU back 
    
    return response.strip()