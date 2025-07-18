from probe_helper import *
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model

import os
os.chdir("/home/ubuntu/myproject/venv")

# import spaces
# warnings.filterwarnings('ignore')

# Questions: 
# what does disk offloading do? Do I need to do it? 
# How do I use TraceDict from baukit to register forward hooks and edit activations? 

#----Load HF access key and model-------#
env_path = find_dotenv() ## Load and get your access token 
HF_TOKEN="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"


llama_model_name = "meta-llama/Llama-3.1-8B-Instruct" 

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=HF_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, 
                                                   token=HF_TOKEN, torch_dtype=torch.float16).to(device)


# model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# llama_70_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
# llama_70_model = AutoModelForCausalLM.from_pretrained(model_id, 
#                                                    token=HF_TOKEN, torch_dtype=torch.float32).to(device)


for name, params in llama_70_model.named_parameters(): 
    print(name)

                         
#-------Preprocess data----------# 
advocacy_corpus = load_data(N=3000)
advocacy_corpus.to_pickle("condensed.pkl")
region_map = {
    "MA": 0,
    "MA ": 0,
    "TX": 1
}
advocacy_corpus['state'] = advocacy_corpus['state'].map(region_map)
advocacy_corpus.loc[(advocacy_corpus['state'] == "MA "), 'state'] = "MA"
advocacy_corpus['state'].value_counts()
advocacy_dict = convert_dataframe_to_json(advocacy_corpus)

#------Run forward pass and extract hidden states------------# 
llama_tokenizer.pad_token = llama_tokenizer.eos_token
representation_dict = extract_hidden_reprs(message_dict=advocacy_dict,
                                           tokenizer=llama_tokenizer,
                                           batch_size=1,
                                           model=llama_model, 
                                           checkpoint_file="reprs_checkpoint_3K.pkl")

# Second save just to be safe 
with open("reprs_checkpoint_complete_save_3K.pkl", "wb") as f: 
   pickle.dump(representation_dict, f)

with open("reprs_checkpoint_complete_save_3K.pkl", 'rb') as f:
        representations = pickle.load(f)
        
torch.cuda.empty_cache()
        
len(representations)
ground_truth_dict = {l['index']: l['state'] for l in advocacy_dict[0:len(representations)]}
economic_frame_ground_truth_dict = {l['index']: l['economic_bin'] for l in advocacy_dict[0:len(representations)]}
capacity_frame_ground_truth_dict = {l['index']: l['capacity_and_resources_bin'] for l in advocacy_dict[0:len(representations)]}
moral_frame_ground_truth_dict = {l['index']: l['morality_bin'] for l in advocacy_dict[0:len(representations)]}
fairness_frame_ground_truth_dict = {l['index']: l['fairness_and_equality_bin'] for l in advocacy_dict[0:len(representations)]}
legality_frame_ground_truth_dict = {l['index']: l['legality_cons_juris_bin'] for l in advocacy_dict[0:len(representations)]}
policy_frame_ground_truth_dict = {l['index']: l['policy_prescription_eval_bin'] for l in advocacy_dict[0:len(representations)]}
crimes_frame_ground_truth_dict = {l['index']: l['crimes_and_punishment_bin'] for l in advocacy_dict[0:len(representations)]}
security_frame_ground_truth_dict = {l['index']: l['security_and_defense_bin'] for l in advocacy_dict[0:len(representations)]}
quality_of_life_frame_ground_truth_dict =  {l['index']: l['quality_of_life_bin'] for l in advocacy_dict[0:len(representations)]}
political_frame_ground_truth_dict =  {l['index']: l['political_bin'] for l in advocacy_dict[0:len(representations)]}
cultural_frame_ground_truth_dict =  {l['index']: l['cultural_identity_bin'] for l in advocacy_dict[0:len(representations)]}
public_op_frame_ground_truth_dict =  {l['index']: l['public_opinion_bin'] for l in advocacy_dict[0:len(representations)]}
health_safety_frame_ground_truth_dict =  {l['index']: l['health_and_safety_bin'] for l in advocacy_dict[0:len(representations)]}
external_reg_frame_ground_truth_dict = {l['index']: l['external_regulation_bin'] for l in advocacy_dict[0:len(representations)]}

df_dict = split_reprs_to_dfs(representations) ## Construct n probe datasets from hidden representations

train_probe(layer_dfs=df_dict, 
            labels=ground_truth_dict, 
            os_dir = "trial_result",
            out_file="clfres_new", 
            cv=5, 
            save_probe=True)

train_probe(layer_dfs=df_dict, 
            labels=economic_frame_ground_truth_dict, 
            os_dir="trial_results_econ", 
            out_file="clfecon_2_new", 
            cv=5, 
            save_probe=True)

train_probe(layer_dfs=df_dict, 
            labels=fairness_frame_ground_truth_dict, 
            os_dir="trial_results_fair", 
            out_file="clffair_2_new", 
            cv=5, 
            save_probe=True)

train_probe(layer_dfs=df_dict, 
            labels=health_safety_frame_ground_truth_dict, 
            os_dir="trial_results_health", 
            out_file="clfhealth_2_new", 
            cv=5, 
            save_probe=True)



best_models, layer_names  = evaluate_probe_classifiers(5, file_dir="trial_result")
out, layer_names = construct_multi_steer_vector(file_dir="trial_result", module=llama_model, top_k=3, alpha=2, norm=True)
hook = intervene_activations(out)

## Not interventions 



prompt_0 = "Eating disorders are a serious public health issue because"
prompt_0_a = "Eating disorders are a serious public health challenge because"
prompt_0_b = "State lawmakers should pass bills to restrict or make it harder for children for using over-the-counter weight-loss and muscle-building supplements because"

prompt_1 = "I think protecting children from using over-the-counter weight-loss supplements, diet pills, muscle-building supplements is important because..."
prompt_1_a = "Some harms of over-the-counter weight-loss supplements, diet pills, and muscle-building supplements include..."

## Solicit model knowledge 
prompt_2 = """
You are an advocacy communications expert.  Generate a compelling, informative, and resonant advocacy message for Texas state lawmakers and other key stakeholders on why it is important to pass a bill to impose tax on over-the-counter weight-loss supplements, diet pills, and muscle-building supplements away from children.
"""
## perform style transfer of type 1 message 
prompt_2_wo_support_materials = """
You are a communications strategist at a top-tier public affairs agency specializing in persuasive messaging for health and safety advocacy campaigns. Your job is to localize and adapt messages for distinct cultural, political, and regional audiences across the US.
We are the Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED), a public health advocacy group. We are launching a critical public health campaign in Texas focused on protecting minors from the harms of over-the-counter weight-loss and muscle-building supplements. Our proposed policies include age-gating mechanisms, such as point-of-sale restrictions and excise taxes, to prevent youth access.
Please generate an advocacy message tailored for Texas. The instructions are to 
Tailor the tone, word choice, and rhetorical style to align with Texan values
Reflect a broad and inclusive regional vernacular 
Make the message resonant and factually consistent and remain grouned in evidence. 
"""

prompt_2_w_support_material = """
You are a communications strategist at a top-tier public affairs agency specializing in persuasive messaging for health and safety advocacy campaigns. Your job is to localize and adapt messages for distinct cultural, political, and regional audiences across the US.
We are the Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED), a public health advocacy group. We are launching a critical public health campaign in Texas focused on protecting minors from the harms of over-the-counter weight-loss and muscle-building supplements. Our proposed policies include age-gating mechanisms, such as point-of-sale restrictions and excise taxes, to prevent youth access.
We have a working message tailored for Massachusetts audiences. Please help us adapt this message for Texas. In doing so:
Tailor the tone, word choice, and rhetorical style to align with Texan values
Reflect a broad and inclusive regional vernacular
Make the message resonant, factually consistent with the resources we share with you while remaining grounded in evidence-based public health concerns.
Your task:
 Carefully rewrite the message (provided below) so that it feels like it was written by and for Texans, while keeping the core policy goal intact.
Protecting young people from dangerous weight loss and muscle-building products
Over-the-counter diet pills and muscle-building supplements are common and widely available to consumers of all ages, with one in five women and one in 10 men reporting ever using these products. While these dietary supplements often claim to promote weight loss or muscle building – many products are sold without any scientific evidence of their safety or effectiveness and are inadequately regulated by the U.S. Food and Drug Administration (FDA). Even more alarming, these deceptive products can be sold to consumers of any age without restriction, so any child can go to their local grocery store, convenience store, or gym or go online to buy these hazardous products. For these reasons, the American Academy of Pediatrics has released two reports strongly cautioning against teens using diet pills and muscle-building supplements for any reason.
Research has documented dire results among users of these products, including liver damage and even death. These products also worsen health inequities as Latinx teens are 40% more likely to use over-the-counter diet pills than white teens. In addition, they are prospectively linked with a higher risk of eating disorders diagnosis and illicit anabolic steroid use. Adolescent and young adult women who use over-the-counter diet pills are 4 to 6 times more likely than their peers to be diagnosed with an eating disorder within several years. And young men and young women who start using muscle-building supplements are 2-5 times more likely than their peers to go on to use illicit anabolic steroids or similar harmful substances.
Lawmakers from California, Massachusetts, New Jersey, and Virginia have introduced legislation to protect child health and keep these dangerous products #OutofKidsHands.
Thank you to the senators, representatives, delegates, and assemblymembers for being champions of bills to protect young people from dangerous weight loss and muscle-building supplements. For more updates on each states’ legislation, check out our Out of Kids’ Hands Campaign pages below.

Use these facts and resources to construct an advocacy message targeted to lawmakers to urge them to support this policy: 
"""

prompt_2_with_support_material_2 = """
You are a communications strategist at a top-tier public affairs agency specializing in persuasive messaging for health and safety advocacy campaigns. Your job is to localize and adapt messages for distinct cultural, political, and regional audiences across the US.
We are the Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED), a public health advocacy group. We are launching a critical public health campaign in Texas focused on protecting minors from the harms of over-the-counter weight-loss and muscle-building supplements. Our proposed policies include age-gating mechanisms, such as point-of-sale restrictions and excise taxes, to prevent youth access.
We have a working message tailored for Massachusetts audiences. Please help us adapt this message for Texas. In doing so:
Tailor the tone, word choice, and rhetorical style to align with Texan values
Reflect a broad and inclusive regional vernacular
Make the message resonant, factually consistent with the resources we share with you while remaining grounded in evidence-based public health concerns.
Your task:
Carefully rewrite the message (provided below) so that it feels like it was written by and for Texans, while keeping the core policy goal intact.

## WHAT IS THE PROBLEM? 
Eating disorders are a serious public health problem affecting youth and adults of all races, ages, and genders. In recent years, research has illuminated significant health disparities in eating disorders: girls report more eating disorder symptoms than do boys; sexual minority and transgender youth are likelier to develop eating disorders than their heterosexual and cisgender counterparts; youth of color are equally likely as white youth to develop eating disorders but less likely to access treatment.Eating disorders are associated with a number of serious health risks including osteoporosis and heart disease. Eating disorders are diagnosed based on a number of criteria, including the presence of what clinicians call unhealthy weight control behaviors (UWCBs). These behaviors can constitute either a symptom or a risk factor for eating disorders, depending on a person’s other behaviors. One UWCB of particular concern is the use of pills or powders to lose weight or build muscle, which are often sold as dietary supplements. Although they are sold alongside multivitamins and other supplements largely regarded as safe, these products often contain unlisted, illegal pharmaceutical ingredients that pose serious risks. Under the Dietary Supplement Health and Education Act of 1994 (DSHEA), the U.S. Food and Drug Administration (FDA) does not have the authority to require proof of safety or efficacy prior to the sale of these products. While some voluntary certifications exist, there is no guarantee that a supplement contains what the label says it does. Supplements sold for weight loss and muscle-building have been found to contain substances including untested designer amphetamine analogues, psychotropic drugs, and the active ingredient from the failed weight-loss drug Meridia, which was pulled from the market in 2010. These products have been linked to outbreaks of liver injury, some severe enough to require transplantation, and have even caused several high-profile deaths in recent years. While DSHEA does grant the FDA the power to test products on the market and initiate recalls, this is not an effective means of protecting the public: One recent study found that two-thirds of recalled supplements still contained contaminants six months after the recalls were initiated. Despite the harms these products can cause, the perception of risk associated with them is still low, and the U.S. market is estimated to exceed $40 billion. Given the severity and scope of this problem, policy intervention is warranted at the state and local level. 

## WHAT CAN WE DO?
 States and municipalities have a number of policy tools at their disposal that can reduce the threats these products pose. Taxation and age restrictions are two evidence-based public health policy strategies that have been used in a number of contexts to reduce youth access to dangerous products, most notably cigarettes and alcohol.  While studies have found that youth may be particularly influenced by taxation, it is often used as a targeted strategy. In the cases of both alcohol and tobacco, taxation has been found to drive down overall consumption in adults as well as youth.  Although supplements for weight loss and muscle-building are risky for adults as well as children, adults are better able than children to assess these risks and make informed decisions concerning supplement use. By contrast, age limits are specific to youth and have been demonstrated to reduce alcohol and tobacco consumption in adolescents when appropriately enforced. Like with alcohol and tobacco, supplements for weight loss and muscle-building can be kept behind a pharmacy counter or in a locked display case; this may have the added benefit of bolstering risk perception. By emphasizing the need to protect youth, imposing age restrictions may serve as an attractive opportunity for businesses to demonstrate corporate social responsibility by engaging as partners in the development and implementation of regulations regarding the display and sale of supplements. STRIPED encourages policymakers and advocates to pursue age restrictions as an evidence-based, politically feasible strategy. 
## WHAT’S HAPPENING IN MA? 
A bill has been introduced by Rep. Kay Khan (D-Newton) during the 191st General Court with input from STRIPED. If passed, H.1942, An Act Protecting Children From Harmful Diet Pills and Muscle-Building Supplements, would: • Restrict the sale of diet pills and supplements sold for weight loss and muscle-building to adults 18 years and over only • Mandate that such products be kept behind a counter or otherwise inaccessible to minors in order to facilitate enforcement of the age restriction • Require the placement of signs alerting consumers to the dangers associated with these products • Direct the Department of Public Health to develop criteria for determining which products are included
Fact Sheet

## The Request 
The Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED) urges state leaders to protect young people from the dangers of diet pills and muscle-building supplements. A new bill aims to address this issue. If passed, this bill will ban the sale of diet pills and muscle-building supplements to minors under 18 years old and will move these products behind the counter, requiring consumers to request them directly from a pharmacist, manager, or other store supervisory personnel. 

## The Problem 
• Our youth are at risk! Diet pills and muscle-building supplements are linked with eating disorders as well as body dysmorphic disorder. More than 30 percent of children and adolescents take dietary supplements on a regular basis,1 and 11%1of teens report ever using dietary supplements for weight loss.
 • Dangerous products. Dietary supplements sold for weight loss and muscle building are associated with serious health risks and side effects including organ failure, testicular cancer, heart attack, stroke, and even death. Some supplements are adulterated with illegal substances such as steroids, prescription pharmaceuticals, and heavy metals.The American Academy of Pediatrics has released reports stating that teens should never use diet pills or muscle-building supplements.
• Insufficient regulation of dietary supplements by the FDA. Supplements are taken off shelves by the FDA only after reports of serious injury or death. 
The attorneys general of 14 states joined in a letter to the U.S. Congress seeking a federal investigation into the dietary supplements industry. But our youth need greater protection now. 

## Steps Your State Can Take to Protect Its Youth 
1. EXCISE TAX FOR MINORS. Implement a tax to help reduce the sale of these products to children. Due to their developmental stage, youth may be unable to weigh the harms linked with these products. 
2. MOVE PRODUCTS BEHIND THE COUNTER. Moving diet pills and muscle-building supplements from the shelves to behind the counter will ensure that consumers will first speak with a pharmacist, manager, or other store supervisory personnel. 
3. URGE THE ATTORNEY GENERAL. Your state legislature can urge the State Attorney General to enforce consumer protection statutes that prohibit unfair or deceptive advertising of diet pills and muscle building supplements. 
4. EDUCATE CONSUMERS. Departments of Public Health can educate consumers about the health risks associated with dietary supplements sold for weight loss and muscle building, as well as the risks associated with misuse and abuse of over-the-counter diet pills.

Use these facts and resources to construct an advocacy message targeted to lawmakers to urge them to support this policy: 
"""

prompt_3_a = """
You are a communication strategist and your goal is to localize and adapt messages for distinct cultural, political, and regional audience while maintaining a professional tone.
We are the Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED), a public health advocacy group. We are launching a critical public health campaign in Texas focused on protecting minors from the harms of over-the-counter weight-loss and muscle-building supplements. Our proposed policies include age-gating mechanisms, such as point-of-sale restrictions and excise taxes, to prevent youth access.
We have a working message tailored for Massachusetts audiences. Please help us adapt this message for Texas. In doing so:
Tailor the tone, word choice, and rhetorical style to align with Texan values.
"""
prompt_3_b = """
You are a communications strategist at a top-tier public affairs agency specializing in persuasive messaging for health and safety advocacy campaigns. Your job is to localize and adapt messages for distinct cultural, political, and regional audiences across the US.
We are the Strategic Training Initiative for the Prevention of Eating Disorders (STRIPED), a public health advocacy group. We are launching a critical public health campaign in Texas focused on protecting minors from the harms of over-the-counter weight-loss and muscle-building supplements. Our proposed policies include age-gating mechanisms, such as point-of-sale restrictions and excise taxes, to prevent youth access.
We have a working message tailored for Massachusetts audiences. Please help us adapt this message for Texas. In doing so:
Tailor the tone, word choice, and rhetorical style to align with Texan values
Reflect a broad and inclusive regional vernacular
Make the message resonant, factually consistent with the resources we share with you while remaining grounded in evidence-based public health concerns.
Your task:
Carefully rewrite the message (provided below) so that it feels like it was written by and for Texans, while keeping the core policy goal intact.
Protecting young people from dangerous weight loss and muscle-building products
Over-the-counter diet pills and muscle-building supplements are common and widely available to consumers of all ages, with one in five women and one in 10 men reporting ever using these products. While these dietary supplements often claim to promote weight loss or muscle building – many products are sold without any scientific evidence of their safety or effectiveness and are inadequately regulated by the U.S. Food and Drug Administration (FDA). Even more alarming, these deceptive products can be sold to consumers of any age without restriction, so any child can go to their local grocery store, convenience store, or gym or go online to buy these hazardous products. For these reasons, the American Academy of Pediatrics has released two reports strongly cautioning against teens using diet pills and muscle-building supplements for any reason.
Research has documented dire results among users of these products, including liver damage and even death. These products also worsen health inequities as Latinx teens are 40% more likely to use over-the-counter diet pills than white teens. In addition, they are prospectively linked with a higher risk of eating disorders diagnosis and illicit anabolic steroid use. Adolescent and young adult women who use over-the-counter diet pills are 4 to 6 times more likely than their peers to be diagnosed with an eating disorder within several years. And young men and young women who start using muscle-building supplements are 2-5 times more likely than their peers to go on to use illicit anabolic steroids or similar harmful substances.
Lawmakers from California, Massachusetts, New Jersey, and Virginia have introduced legislation to protect child health and keep these dangerous products #OutofKidsHands.
Thank you to the senators, representatives, delegates, and assemblymembers for being champions of bills to protect young people from dangerous weight loss and muscle-building supplements. For more updates on each states’ legislation, check out our Out of Kids’ Hands Campaign pages below.

Use these facts and resources to construct an advocacy message targeted to lawmakers to urge them to support this policy: 
"""

## Advocacy overview 
prompt_4_a = """
You are a communications strategist at a top-tier public affairs agency specializing in persuasive messaging for health and safety advocacy campaigns. Your job is to localize and adapt messages for distinct cultural, political, and regional audiences specifically lawmakers and community groups in Texas. Please adapt the following message:

Dear Honorable Chairs and Vice Chairs of the Joint Committee on Health Care Financing:

I am writing on behalf of the Massachusetts Chapter of the American Academy of Pediatrics (MCAAP) in support of H.2215/S.1465, which received a favorable report from the Joint Committee on Public Health. The MCAAP represents more than 1,600 primary care
pediatricians, pediatric medical subspecialists, pediatric surgical specialists, pediatric
residents and medical students. Our members are dedicated to improving the quality of life
for children by providing quality health care and advocating for them and their families.
Dietary supplements that claim to promote weight loss or muscle building are sold to the
public without any scientific evidence supporting their efficacy or safety. For the most part,
they are not regulated by the US Food and Drug Administration (FDA), leaving consumer
safety at risk.

Children and adolescents that take dietary supplements regularly without a doctor’s advice risk dire health consequences, including liver damage from some supplements promising
weight loss and muscle building. Research indicates that approximately 7% of children and
adolescents take two or more dietary supplements on a regular basis.
H.2215/S.1465 places reasonable restraints on an unregulated industry toward the sale of
dietary supplements to children and adolescents. It would ban the sale of over the counter diet
pills and dietary supplements for weight loss or muscle building to anyone under the age of
18. It also would require the posting at retail establishments selling these products a notice
that certain over-the-counter diet pills, or dietary supplements for weight loss or muscle
building are known to cause gastrointestinal impairment, tachycardia, hypertension,
myocardial infarction, stroke, severe liver injury sometimes requiring transplant or leading to
death, organ failure, other serious injury, and death.

The MCAAP urges the Committee to protect young people from these dangerous products and give H.2215/S.1465 a favorable report.

Use these facts and resources to construct an advocacy message targeted to lawmakers to urge them to support this policy: 
"""


response = pd.DataFrame(columns=['prompt','tokens', 'no_interv_t1', 'no_interv_t3', 'no_interv_t7','interv_1','interv_2', 'interv_4', 'econ_interv_1', 'econ_interv_2', 'econ_interv_4', 'health_interv_1', 'health_interv_2', 'health_interv_4'])

prompt_collections = [prompt_0] + [prompt_0_a] + [prompt_0_b] + [prompt_1] + [prompt_1_a] + [prompt_2] + [prompt_2_wo_support_materials] + [prompt_2_w_support_material] + [prompt_2_with_support_material_2] + [prompt_3_b] + [prompt_4_a]
response['prompt'] = pd.Series(prompt_collections)
response['tokens'] = pd.Series([40, 40, 40, 80, 80, 500, 500, 600, 600, 600, 600])

for idx, row in response.iterrows(): 
    pr = row['prompt']
    max_new_tokens = row['tokens'] 
    no_intervene_message_temp_1 = generate_messages(module=llama_model, 
                                           tokenizer=llama_tokenizer, 
                                           input_prompt=pr, 
                                           device='cuda', 
                                           do_sample=True, 
                                           temperature=0.1, 
                                           max_new_tokens=max_new_tokens)
    no_intervene_message_temp_3 = generate_messages(module=llama_model, 
                                           tokenizer=llama_tokenizer, 
                                           input_prompt=pr, 
                                           device='cuda', 
                                           do_sample=True, 
                                           temperature=0.3, 
                                           max_new_tokens=max_new_tokens)
    no_intervene_message_temp_7 = generate_messages(module=llama_model, 
                                           tokenizer=llama_tokenizer, 
                                           input_prompt=pr, 
                                           device='cuda', 
                                           do_sample=True, 
                                           temperature=0.7, 
                                           max_new_tokens=max_new_tokens)
    response.loc[idx, 'no_interv_t1'] = no_intervene_message_temp_1
    response.loc[idx, 'no_interv_t3'] = no_intervene_message_temp_3
    response.loc[idx, 'no_interv_t7'] = no_intervene_message_temp_7

response.to_csv('result/response_results.csv')


for idx, row in response.iterrows(): 
    pr = row['prompt'] 
    max_new_tokens =row['tokens']
    interv_message_1 = intervene_top_k_attention(
            module=llama_model,
            file_dir="trial_result", 
            top_k=4, 
            alpha=1,
            tokenizer=llama_tokenizer,
            input_prompt=pr, 
            norm=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            temperature=0.3, 
            do_sample=True,
            max_new_tokens=max_new_tokens
    )
    
    interv_message_2 = intervene_top_k_attention(
            module=llama_model,
            file_dir="trial_result", 
            top_k=4, 
            alpha=2,
            tokenizer=llama_tokenizer,
            input_prompt=pr, 
            norm=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            temperature=0.3, 
            do_sample=True,
            max_new_tokens=max_new_tokens
    )
    
    interv_message_4 = intervene_top_k_attention(
            module=llama_model,
            file_dir="trial_result", 
            top_k=4, 
            alpha=4,
            tokenizer=llama_tokenizer,
            input_prompt=pr, 
            norm=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            temperature=0.3, 
            do_sample=True,
            max_new_tokens=max_new_tokens
    )
    
    
    response.loc[idx, 'interv_1'] = interv_message_1
    response.loc[idx, 'interv_2'] = interv_message_2
    response.loc[idx, 'interv_4'] = interv_message_4
    
response.loc[0, 'no_interv_t1']
response.loc[1, 'no_interv_t3']
response.loc[5, 'no_interv_t7']    



for idx, row in response.iterrows(): 
    pr = row['prompt']
    max_new_tokens = row['tokens']
    no_interv_message = generate_messages(module=llama_model, 
                                        tokenizer=llama_tokenizer,
                                        input_prompt=pr,
                                        device="cuda",
                                        do_sample=True,
                                        temperature=0.7, 
                                        max_new_tokens=max_new_tokens)
    
    response.loc[idx, "no_interv"] = no_interv_message
    
    interv_message_1 = intervene_top_k_attention(
            module=llama_model,
            file_dir="trial_result", 
            top_k=3, 
            alpha=1,
            tokenizer=llama_tokenizer,
            input_prompt=pr, 
            norm=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            temperature=0.3, 
            do_sample=True,
            max_new_tokens=max_new_tokens
    )
    

    
    response.loc[idx, "interv_1"] = interv_message_1


no_interv_message_list = []
interv_message_list = []
interv_message_alpha_4_list = []
interv_message_alpha_6_list = []
interv_message_economic_frames = []

for pr in tqdm([prompt_1, prompt_2_wo_support_materials, prompt_2_w_support_material, prompt_2_with_support_material_2]): 
    no_interv_message = generate_messages(module=llama_model, 
                                        tokenizer=llama_tokenizer,
                                        input_prompt=pr,
                                        device="cuda",
                                        do_sample=True,
                                        temperature=0.7, 
                                        max_new_tokens=600)
    interv_message = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=2,
        tokenizer=llama_tokenizer,
        input_prompt=pr, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
    interv_message_alpha_4 = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=4,
        tokenizer=llama_tokenizer,
        input_prompt=pr, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
    interv_message_alpha_6 = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=4,
        tokenizer=llama_tokenizer,
        input_prompt=pr, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
    interv_message_econ = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_results_econ", 
        top_k=5, 
        alpha=2,
        tokenizer=llama_tokenizer,
        input_prompt=pr, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
    
    no_interv_message_list.append(no_interv_message)
    interv_message_list.append(interv_message)
    interv_message_alpha_4_list.append(interv_message_alpha_4)
    interv_message_alpha_6_list.append(interv_message_alpha_6)
    interv_message_economic_frames.append(interv_message_econ)


print(no_interv_message_list)
print(interv_message_list)
print(interv_message_alpha_4_list)
print(interv_message_alpha_6_list)
print(interv_message_economic_frames)


complete_response = []

prompt = """
“Create a persuasive advocacy message that raises awareness about the dangers of youth using over-the-counter weight-loss and muscle-building supplements. The message should be informative, compelling, and resonant for a general audience in Texas.”
"""
no_interv_message = generate_messages(module=llama_model, 
                                        tokenizer=llama_tokenizer,
                                        input_prompt=prompt,
                                        device="cuda",
                                        do_sample=True,
                                        temperature=0.7, 
                                        max_new_tokens=600)
complete_response.append(no_interv_message)

interv_message = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=2,
        tokenizer=llama_tokenizer,
        input_prompt=prompt, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
complete_response.append(interv_message)

interv_message_alpha_4 = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=4,
        tokenizer=llama_tokenizer,
        input_prompt=prompt, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
complete_response.append(interv_message_alpha_4)


interv_message_alpha_6 = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_result", 
        top_k=5, 
        alpha=6,
        tokenizer=llama_tokenizer,
        input_prompt=prompt, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )

interv_message_econ = intervene_top_k_attention(
        module=llama_model,
        file_dir="trial_results_econ", 
        top_k=5, 
        alpha=2,
        tokenizer=llama_tokenizer,
        input_prompt=prompt, 
        norm=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7, 
        do_sample=True,
        max_new_tokens=600
    )
    

complete_response.append(interv_message_alpha_6)
complete_response.append(interv_message_econ)

responses = intervene_top_k_attention(
    module=llama_model, 
    file_dir="trial_result", 
    top_k=5, 
    alpha=1.5,
    tokenizer=llama_tokenizer, 
    input_prompt=prompt, 
    norm=True, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    temperature=0.7, 
    do_sample=True,
    max_new_tokens=500
)

for name, params in llama_model.named_modules(): 
    print(name)
prompt = """
“Create a persuasive advocacy message that raises awareness about the dangers of youth using over-the-counter weight-loss and muscle-building supplements. The message should be informative, compelling, and appropriate for a general audience in Texas.”
"""
inputs = llama_tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
with torch.no_grad(): 
    output_ids = llama_model.generate(
        **inputs, 
        max_new_tokens=60,
        temperature=0.4, 
        do_sample=True
    )

output_words = llama_tokenizer.decode(output_ids[0], skip_special_token=True)
print(output_words)
