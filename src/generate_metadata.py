import json
import csv
import pandas as pd
from pyinflect import getAllInflections, getInflection
import inflect
import ast

inf = inflect.engine()

# read the base csv files for generating the frames
df_aevery = pd.read_csv("dataset/templates/a_every.csv")
df_donkeycond = pd.read_csv("dataset/templates/donkey_cond.csv")
df_neg_with_altv = pd.read_csv('dataset/templates/donkey_cond_neg.csv')
df_disjunction_updated = pd.read_csv('dataset/templates/disjunction.csv')

def get_exp1a_sent(quant, gender, row, scope):
    """A vs. Every"""
    if gender == 'm':
        pronoun = 'He' 
    elif gender == 'f':
        pronoun = 'She'
    else:
        pronoun = 'They'
    quant_to_quantifier = {
        "ext": "A",
        "uni": "Every",
        "no": "No",
    }

    if scope == "outScope":
        if quant == "ext":
            return f"{inf.a(row['noun1']).capitalize()} {row['verb1_PP']}. {pronoun} {row['verb2']} {row['verb2_PP']}."
        else:
            return f"{quant_to_quantifier[quant]} {row['noun1']} {row['verb1_PP']}. {pronoun} {row['verb2']} {row['verb2_PP']}."
    elif scope == "inScope":
        if quant == "ext":
            return f"{inf.a(row['noun1']).capitalize()} {row['verb1_PP']} before {pronoun.lower()} {row['verb2']} {row['verb2_PP']}."
        else:
            return f"{quant_to_quantifier[quant]} {row['noun1']} {row['verb1_PP']} before {pronoun.lower()} {row['verb2']} {row['verb2_PP']}."
    else:
        raise ValueError("Invalid scope specification.")

def generate_exp1a_data(df):
    """
    batch generate the frames for exp1a
    Expected 32 * 3 * 3 * 3 = 864 items
    """
    for i in range(len(df)):
        for gender in ['m', 'f', 'p']:
            for quant in ['ext', 'uni', 'no']:
                for scope in ["inScope", "outScope"]:
                    dic = {'frame':i,
                           'gender':gender,
                           'sent':get_exp1a_sent(quant, gender, df.iloc[i], scope),
                           'sentence_type':quant,
                           'scope':scope,}
                    with open('dataset/exp1a_metadata.jsonl', 'a') as f:
                        f.write(json.dumps(dic) + '\n')

def get_exp1b_sent(j, gender, cont, row, subj_type, quant, scope):
    """If, whenever, every using donkey_conditional
    - Conditional: If {John, the farmer} owns a cow, he beats it.
    - Whenever: Whenever {John, the farmer} owns a cow, he beats it.
    - Existential: {John, the farmer} owns a cow, and he beats it.
    - *No change to `every`*;

    Potentially good baselines for bad sentences (to show that they are really bad);
    - Conditional_inScope: If {John, the farmer} owns a cow, he beats it and it is brown.
    - vs. Conditional_bad:  If {John, the farmer} owns a cow, he beats it. It is brown;
    - *** same for whenever and existential;
    """
    pronoun_subj = 'he' if gender == 'm' else 'she'

    if subj_type == "name":
        subj = 'John' if gender == 'm' else 'Mary'
    else:
        subj = f"the {row['noun1']}"
    
    TEMPLATE_CONTEXT = {
        "if": f"If {subj} {getInflection(row['verb1'], tag='VBZ')[0]} {inf.a(row['noun2'])}, {pronoun_subj} {getInflection(row['verb2'], tag='VBZ')[0]} it.",
        "whenever": f"Whenever {subj} {getInflection(row['verb1'], tag='VBZ')[0]} {inf.a(row['noun2'])}, {pronoun_subj} {getInflection(row['verb2'], tag='VBZ')[0]} it.",
        "every": f"Every {row['noun1']} who {getInflection(row['verb1'], tag='VBZ')[0]} {inf.a(row['noun2'])} {getInflection(row['verb2'], tag='VBZ')[0]} it.",
        "existential": f"{subj.capitalize()} {getInflection(row['verb1'], tag='VBZ')[0]} {inf.a(row['noun2'])}, and {pronoun_subj} {getInflection(row['verb2'], tag='VBZ')[0]} it."
    }
    context_sen = TEMPLATE_CONTEXT[quant]

    TEMPLATE_CONT = {'it': f"It is {inf.a(ast.literal_eval(row['continuation1'])[j])} one.",
                    'he': f"{pronoun_subj.capitalize()} also {getInflection(ast.literal_eval(row['continuation2_would'])[j], tag='VBZ')[0]} it."}
    
    cont_sen = TEMPLATE_CONT[cont]

    if scope == "outScope":
        return context_sen + " " + cont_sen
    elif scope == "inScope":
        if cont == "he":
            return f"{context_sen[:-1]} and {' '.join(cont_sen.split(' ')[1:])}"
        elif cont == "it":
            return f"{context_sen[:-1]} and {cont_sen[0].lower()}{cont_sen[1:]}"



def generate_exp1b_data(df_donkey):
    """
    We are expecting (32 * 5 * 3 * 2 * 2 * 2 + 32 * 5 * 1 * 2 * 2 )*2 = 8960   items
    """
    for i in range(len(df_donkey)):
        for j in range(5):
            for subj_type in ['name', 'def']:
                for gender in ['m', 'f']:
                    for cont in ['he', 'it']:
                        for quant in ['if', 'whenever', 'every', 'existential']:
                            if quant == "every" and subj_type == "def":
                                continue # the subject is always "every..." for every sentences
                            else:
                                for scope in ["inScope", "outScope"]:
                                    dic = {'frame':i,
                                           'pred':j,
                                        'subj':subj_type,
                                        'gender':gender,
                                        'cont':cont,
                                        'scope':scope,
                                        'sent':get_exp1b_sent(j, gender, cont, df_donkey.iloc[i], subj_type, quant, scope),
                                        'sentence_type':quant,
                                        }
                                    with open('dataset/exp1b_metadata.jsonl', 'a') as f:
                                        f.write(json.dumps(dic) + '\n')
        
def get_exp3_sent(sentence_type, row, pronoun, subj):
    if pronoun == "he":
        name = "John"
    else:
        name = "Mary"

    # subj = def or name
    print(getInflection(row['verb1'], tag='VBN')[0])
    TEMPLATES_def = {
        'disjunction_or' : f"There was no {row['noun2']}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by the {row['noun1']}.",

        'disjunction_eitheror' : f"Either there was no {row['noun2']}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by the {row['noun1']}.",

        'disjunction_eitheror_pos': f"Either there was {inf.a(row['noun2'])}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by the {row['noun1']}.",
         
        'conjunction' : f"There was no {row['noun2']}, and it was {getInflection(row['verb1'], tag='VBN')[0]} by the {row['noun1']}.",

    }
    TEMPLATES_name = {
        'disjunction_or' : f"There was no {row['noun2']}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by {name}.",

        'disjunction_eitheror' : f"Either there was no {row['noun2']}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by {name}.",

        'disjunction_eitheror_pos': f"Either there was {inf.a(row['noun2'])}, or it was {getInflection(row['verb1'], tag='VBN')[0]} by {name}.",
         
        'conjunction' : f"There was no {row['noun2']}, and it was {getInflection(row['verb1'], tag='VBN')[0]} by {name}.",

    }
    
    if subj == "def":
        return TEMPLATES_def[sentence_type]
    elif subj == "name":
        return TEMPLATES_name[sentence_type]
    else:
        raise ValueError("Invalid subject.")
    

def get_exp2_sent(sentence_type, row, pronoun, subj):
    if pronoun == "he":
        name = "John"
    else:
        name = "Mary"

    curr_pred = row["predicate"]

    altv = row['alt_verb'].split(",")[0]
    # subj = def or name
    TEMPLATES_def = {        
        'DNE' : f"It was not the case that the {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. It was just {curr_pred}.",
        
        'DNE_infact' : f"It was not the case that the {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was just {curr_pred}.",
        
        'negation2' : f"The {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. It was {curr_pred}.",
         
        'negation2_just_infact' : f"The {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was just {curr_pred}.",

        'negation2_just' : f"The {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. It was just {curr_pred}.",
         
        'negation2_infact' : f"The {row['noun1']} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was {curr_pred}.",

        'existential' : f"The {row['noun1']} {getInflection(row['verb1'], tag='VBD')[0]} {inf.a(row['noun2'])}. It was {curr_pred}.",

        'existential_infact' : f"The {row['noun1']} {getInflection(row['verb1'], tag='VBD')[0]} {inf.a(row['noun2'])}. In fact, it was {curr_pred}.",
    
    }
    TEMPLATES_name = {
        'DNE' : f"It was not the case that {name} didn't {row['verb1']} {inf.a(row['noun2'])}. It was just {curr_pred}.",

        'DNE_infact' : f"It was not the case that {name} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was just {curr_pred}.",
        
        'negation2' : f"{name} didn't {row['verb1']} {inf.a(row['noun2'])}. It was {curr_pred}.",

        'negation2_infact' : f"{name} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was {curr_pred}.",

        'negation2_just' : f"{name} didn't {row['verb1']} {inf.a(row['noun2'])}. It was just {curr_pred}.",

        'negation2_just_infact' : f"{name} didn't {row['verb1']} {inf.a(row['noun2'])}. In fact, it was just {curr_pred}.",

        'existential' : f"{name} {getInflection(row['verb1'], tag='VBD')[0]} {inf.a(row['noun2'])}. It was {curr_pred}.",
        
        'existential_infact' : f"{name} {getInflection(row['verb1'], tag='VBD')[0]} {inf.a(row['noun2'])}. In fact, it was {curr_pred}.",
    }
    
    if subj == "def":
        return TEMPLATES_def[sentence_type]
    elif subj == "name":
        return TEMPLATES_name[sentence_type]
    else:
        raise ValueError("Invalid subject.")
    
def generate_exp2_data(df):
    # should generate a jsonl file that is 32 * 3 * 6 = 576
    for subj in ["def", "name"]:
        if subj == "def":
            for idx, row in df.iterrows():
                
                for sentence_type in ['DNE', 'DNE_infact', 'negation2', 'negation2_infact', 'negation2_just', 'negation2_just_infact', 'existential', 'existential_infact']:
                
                
                    sen = get_exp2_sent(sentence_type, row, "NA", subj)
                    # print(sentence_type, sen)
                    

                    dic = {'frame': f'{idx}',
                            'gender': "NA",
                            'sent': sen,
                            'subject': subj,
                            'sentence_type': sentence_type}
                        
                    with open(f'dataset/exp2_metadata.jsonl', 'a') as output:
                        output.write(json.dumps(dic) + '\n')
        elif subj == "name":
            for idx, row in df.iterrows():
                for gender in ['m','f']:
                    pronoun = 'he' if gender == 'm' else 'she'
                    
                    for sentence_type in ['DNE', 'DNE_infact', 'negation2', 'negation2_infact',  'negation2_just', 'negation2_just_infact', 'existential', 'existential_infact']:
                    
                    
                        sen = get_exp2_sent(sentence_type, row, pronoun, subj)
                        # print(sentence_type, sen)
                        

                        dic = {'frame': f'{idx}',
                                'gender': gender,
                                'sent': sen,
                                'subject': subj,
                                'sentence_type': sentence_type}
                            
                        with open(f'dataset/exp2_metadata.jsonl', 'a') as output:
                            output.write(json.dumps(dic) + '\n')
    

def generate_exp3_data(df):
    # should generate a jsonl file that is 2 by 32 by 2 by 3 = 384
    for subj in ["def", "name"]:
        for idx, row in df.iterrows():
            for gender in ['m','f']:
                pronoun = 'he' if gender == 'm' else 'she'
                
                for sentence_type in ['disjunction_eitheror', 'disjunction_eitheror_pos', 'disjunction_or', 'conjunction']:
                
                
                    sen = get_exp3_sent(sentence_type, row, pronoun, subj)
                    # print(sentence_type, sen)
                    

                    dic = {'frame': f'{idx}',
                            'gender': gender,
                            'sent': sen,
                            'subject': subj,
                            'sentence_type': sentence_type}
                        
                    with open(f'dataset/exp3_metadata.jsonl', 'a') as output:
                        output.write(json.dumps(dic) + '\n')

if __name__ == "__main__":
    generate_exp1a_data(df_aevery)
    generate_exp1b_data(df_donkeycond)
    generate_exp2_data(df_neg_with_altv)
    generate_exp3_data(df_neg_with_altv)
    generate_exp3_data(df_disjunction_updated)