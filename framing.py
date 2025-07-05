from openai import OpenAI
import os
from tools import chained_ner
from dotenv import find_dotenv, load_dotenv
from typing import Optional, List, Any, Tuple
import random
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm
import time
import json
import ast
        
def construct_prompt(instructions: str, 
                     continuation_texts: str, 
                     few_shot: bool = False,
                     few_shot_texts: Optional[List[str]] = None, 
                     few_shot_answers: Optional[List[int]] = None, 
                     few_shot_num: Optional[int] = 3) -> str:
    """Takes prompt instructions, a question, and optional few-shot examples, randomizes the few-shot examples,
    adds contextual calibration into a fully constructed prompt. 

    Args:
        instructions (str): Prompt 
        few_shot (bool): If prompt provides few shot examples 
        continuation_texts (str): Question
        few_shot_texts (Optional[List[str]], optional): A list of few shot examples. Defaults to None.
        few_shot_answers (Optional[List[int]], optional): A list of few shot answers. Defaults to None.
        few_shot_num (Optional[int], optional): How many few shot examples to use? Defaults to 3.

    Returns:
        full_prompt (str): A fully constructed prompt that can be used as input for language model.
    """
    
    
    prompt = instructions.strip() + "\n\n"
    
    prompt_structure = ChatPromptTemplate.from_template(prompt + "{exemplar_icl}" + "{text}" + "\n\nAnswer:\n")

    if few_shot == True: 
        
        ## Shuffle codes to prevent recency bias (Zhao et. al 2021)
        examples = list(zip(few_shot_texts, few_shot_answers))
        random.shuffle(examples)
        few_shot_texts, few_shot_answers = zip(*examples)
        
        few_shot_examples =""
        
        for i in range(min(few_shot_num, len(few_shot_texts))): 
            few_shot_examples += "\n\n###Examples:\n" + few_shot_texts[i] + "\n\n###Answer:\n" + few_shot_answers[i]

        ## Optional contextual calibration (Zhao et al. 2021)
        few_shot_examples += "\n\n###Examples:\n" + "N/A [MASK] [MASK]" + "\n\n###Answer:\n" + "N/A" + "\n\n"
            
            
    else: 
        few_shot_examples = " "
    
    full_prompt = prompt_structure.format(exemplar_icl=few_shot_examples, text=continuation_texts)
        
    return full_prompt


env_path = find_dotenv()
if load_dotenv(env_path): 
    OPENAI_KEY = os.getenv("OAI_TOKEN")

# Initialize large language model 
client = ChatOpenAI(temperature=0, 
                    model='gpt-4o',
                    openai_api_key=OPENAI_KEY,
                    logprobs=True)

df = pd.read_csv('data/advocacy_corpus.csv')

prompt = """
You are a political and advocacy communications expert who will closely a text to determine 
if any texts employ rich or subtle contextual frames. Frames are messaging tactics directed towards 
the audience to make an aspect of perceived reality more salient. Often, frames are used to promote
problem definition, causal interpretation, moral evaluation, or treatment recommendations for the issue at hand.
Frames can sometimes employed via social or cultural slogans.

Economic: costs, benefits, or other financial implications
Capacity and resources: availability of physical, human
or financial resources, and capacity of current systems
Morality: religious or ethical implications
Fairness and equality: balance or distribution of rights,
responsibilities, and resources
Legality, constitutionality and jurisprudence: rights,
freedoms, and authority of individuals, corporations, and
government
Policy prescription and evaluation: discussion of specific
policies aimed at addressing problems
Crime and punishment: effectiveness and implications of
laws and their enforcement
Security and defense: threats to welfare of the individual,
community, or nation
Health and safety: health care, sanitation, public safety
Quality of life: threats and opportunities for the individual’s wealth, happiness, and well-being
Cultural identity: traditions, customs, or values of a social
group in relation to a policy issue
Public opinion: attitudes and opinions of the general public, including polling and demographics
Political: considerations related to politics and politicians,
including lobbying, elections, and attempts to sway voters
External regulation and reputation: international reputation or foreign policy of the U.S.
Other: any coherent group of frames not covered by the
above categories

Given a text you will read it closely and slowly thinking through each statement. 

```
{{
    "frames": [
        {{
            "type": "Economic",
            "extractive": [all corresponding sentences]
        }}, 
        {{
            "type": "Health and safety", 
            "extractive": [all corresponding sentences]
        }},
}}
```
```
{{
  "frames": [],
  "no_frames_detected": true
}}
```
"""


df['zero_shot_frames_q'] = df['extracted_secondary_texts'].map(lambda x: construct_prompt(instructions=prompt, 
                                                               continuation_texts=x, 
                                                               few_shot=False))

model_responses = []
for idx, row in tqdm(df.iterrows(), total=len(df)): 
    res = client.invoke(row['zero_shot_frames_q'])
    model_responses.append(res)
    
model_responses_2 = []
for idx, row in tqdm(df[785:].iterrows(), total=len(df[785:])):
    res = client.invoke(row['zero_shot_frames_q'])
    model_responses_2.append(res)
    
model_responses_3 = []
for idx, row in tqdm(df[1403:].iterrows(), total=len(df[1403:])):
    res = client.invoke(row['zero_shot_frames_q'])
    model_responses_3.append(res)
    
df['model_res_lang'] = pd.Series(model_responses)
df.loc[785:785 + len(model_responses_2) - 1, 'model_res_lang'] = model_responses_2
df.loc[1403:1403 + len(model_responses_3) - 1, 'model_res_lang'] = model_responses_3

complete_model_res = model_responses + model_responses_2 + model_responses_3

responses = []
for res in complete_model_res: 
    responses.append(res.content)

df['model_responses'] = pd.Series(responses)
df.to_pickle('data/FINAL_DF.pkl')
assert len(df) == len(responses) # Match model responses with the length of dataframe

frame_cols = [
    "economic", 
    "capacity_and_resources", 
    "morality", 
    "fairness_and_equality",
    "legality_cons_juris", 
    "policy_prescription_eval",
    "crimes_and_punishment",
    "security_and_defense", 
    "quality_of_life",
    "health_and_safety",
    "cultural_identity",
    "public_opinion",
    "political",
    "external_regulation",
    "other_frames", 
    "none_detected"
]


df[frame_cols] = None
for i, res in tqdm(enumerate(responses), total=len(responses)): 
    try:
        res_dict = json.loads(res.strip("```").strip("json"))  # Deserialize the object 

        if res_dict.get("no_frames_detected"): 
            df.loc[i, "none_detected"] = 1
        else: 
            frame_types = res_dict['frames']
            df.loc[i, "none_detected"] = 0

            for t in frame_types:
                frame = t['type'].lower().strip()

                if frame in ["economic", "economics"]:
                    df.loc[i, "economic"] = str(t.get("extractive", 1))

                elif frame in ["capacity and resources", "capacity & resources", "resource capacity"]:
                    df.loc[i, "capacity_and_resources"] = str(t.get("extractive", 1))

                elif frame in ["morality", "moral", "ethical concerns"]:
                    df.loc[i, "morality"] = str(t.get("extractive", 1))

                elif frame in ["fairness and equality", "equality and fairness", "equity"]:
                    df.loc[i, "fairness_and_equality"] = str(t.get("extractive", 1))

                elif frame in [
                    "legality, constitutionality, and jurisprudence", "legal, jurisprudence", 
                    "legality", "constitutional law", "jurisprudence"]:
                    df.loc[i, "legality_const_juris"] = str(t.get("extractive", 1))

                elif frame in [
                    "policy prescription and evaluation", "policy recommendation", "policy evaluation"]:
                    df.loc[i, "policy_prescription_eval"] = str(t.get("extractive", 1))

                elif frame in [
                    "crimes and punishment", "crime and punishment", "criminal justice", "punishment"
                ]:
                    df.loc[i, "crimes_and_punishment"] = str(t.get("extractive", 1))

                elif frame in [
                    "security and defense", "national security", "defense and security"
                ]:
                    df.loc[i, "security_and_defense"] = str(t.get("extractive", 1))

                elif frame in [
                    "quality of life", "standard of living", "life quality"
                ]:
                    df.loc[i, "quality_of_life"] = str(t.get("extractive", 1))

                elif frame in [
                    "health and safety", "public health", "health", "safety"
                ]:
                    df.loc[i, "health_and_safety"] = str(t.get("extractive", 1))

                elif frame in [
                    "cultural identity", "identity", "culture"
                ]:
                    df.loc[i, "cultural_identity"] = str(t.get("extractive", 1))

                elif frame in [
                    "public opinion", 
                    "popular opinion", 
                    "voter opinion"
                ]:
                    df.loc[i, "public_opinion"] = str(t.get("extractive", 1))

                elif frame in [
                    "political", 
                    "partisan", 
                    "government", 
                    "politics"
                ]:
                    df.loc[i, "political"] = str(t.get("extractive", 1))

                elif frame in [
                    "external regulation and reputation", 
                    "international regulation", 
                    "reputation"
                ]:
                    df.loc[i, "external_regulation"] = str(t.get("extractive", 1))

                elif frame in [
                    "other", 
                    "miscellaneous", 
                    "misc", 
                    "other frames"
                ]:
                    df.loc[i, "other_frames"] = str(t.get("extractive", 1))

                # If somehow the frame is "none", still mark it explicitly
                elif frame in ["none", "no frames"]:
                    df.loc[i, "none_detected"] = 1
    except: 
        df.loc[i, "none_detected"] = 1
            

def safe_str_to_list(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val  # already a list
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None  # or raise, or return val depending on what you want

def combine_all_frames(row, cols):
    combined = []
    for col in cols:
        val = row[col]
        if isinstance(val, list):
            combined.extend(val)
        elif val is not None:
            # If not list but something else (string?), you can decide how to handle:
            combined.append(val)
    return combined

df = df.loc[(df["none_detected"] != 1)].reset_index(drop=True)
df[frame_cols] = df[frame_cols].map(safe_str_to_list)
df["all_frames"] = df[frame_cols].apply(lambda row: combine_all_frames(row, frame_cols), axis=1)
advocacy_frames_corpus = df.copy().reset_index(names='article_index').explode(column=['all_frames']
                                        ).reset_index(drop=True)


# Create binary variables for what kind of frames the messages are

def add_frame_bins(df, frame_cols, all_frames_col='all_frames'):
    for col in frame_cols:
        bin_col = f"{col}_bin"
        df[bin_col] = df.apply(
            lambda row: int(
                any(sent in (row[all_frames_col] or []) for sent in (row[col] or []))
            ),
            axis=1
        )
    return df

advocacy_frame_corpus = add_frame_bins(advocacy_frames_corpus, frame_cols=frame_cols, all_frames_col='all_frames')
            
advocacy_frame_corpus.to_pickle('data/advocacy_frames_corpus.pkl')
#advocacy_frames_corpus.to_csv('data/advocacy_frames_corpus.csv', index=False)


frame_cols_bins = [f"{col}_bin" for col in frame_cols]
advocacy_frame_corpus[frame_cols_bins].describe().T
"""
                                count      mean       std  min  25%  50%  75%  max
economic_bin                  30901.0  0.350960  0.477278  0.0  0.0  0.0  1.0  1.0
capacity_and_resources_bin    30901.0  0.073687  0.261265  0.0  0.0  0.0  0.0  1.0
morality_bin                  30901.0  0.003495  0.059016  0.0  0.0  0.0  0.0  1.0
fairness_and_equality_bin     30901.0  0.205916  0.404376  0.0  0.0  0.0  0.0  1.0
legality_cons_juris_bin       30901.0  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0
policy_prescription_eval_bin  30901.0  0.375004  0.484132  0.0  0.0  0.0  1.0  1.0
crimes_and_punishment_bin     30901.0  0.086470  0.281061  0.0  0.0  0.0  0.0  1.0
security_and_defense_bin      30901.0  0.015598  0.123917  0.0  0.0  0.0  0.0  1.0
quality_of_life_bin           30901.0  0.015113  0.122004  0.0  0.0  0.0  0.0  1.0
health_and_safety_bin         30901.0  0.391152  0.488016  0.0  0.0  0.0  1.0  1.0
cultural_identity_bin         30901.0  0.053170  0.224376  0.0  0.0  0.0  0.0  1.0
public_opinion_bin            30901.0  0.013948  0.117276  0.0  0.0  0.0  0.0  1.0
political_bin                 30901.0  0.030743  0.172624  0.0  0.0  0.0  0.0  1.0
external_regulation_bin       30901.0  0.000065  0.008045  0.0  0.0  0.0  0.0  1.0
other_frames_bin              30901.0  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0
none_detected_bin             30901.0  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0
"""

# Named entity recognition removal 
advocacy_frame_corpus = advocacy_frame_corpus.dropna(subset=['all_frames'], axis=0).reset_index(drop=True)

advocacy_frame_corpus['all_frames_preprocessed'] = advocacy_frame_corpus['all_frames'].map(lambda row: chained_ner(row) if row else None)


advocacy_frame_corpus.to_pickle("data/dataset_with_frames_annot_prep.pkl")


