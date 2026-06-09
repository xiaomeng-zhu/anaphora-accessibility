"""
Using gpt-4o API for prompting
"""

import openai
import argparse, json, os, random
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(filename="prompt.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser(description="Running inference using Llama models...")
parser.add_argument("--dataset", help="Dataset name: e.g. exp1a")
args = parser.parse_args()

model_name = "gpt-4o"
dataset = args.dataset
print(f"Running {model_name} on {dataset}")

stimuli_path = f"dataset/{dataset}_metadata.jsonl"
print(f"Loading stimuli from {stimuli_path}...")


def load_data(fname):
    data = []
    with open(fname, "r") as jsonl_f:
        for line in jsonl_f:
            ex = json.loads(line)
            data.append(ex)
            # print(ex)
    return data

data = load_data(stimuli_path)
print("Finished loading stimuli")

# Define a function to get a response from GPT-4o
def get_gpt4o_response(model_name, sent1, sent2, temp):
    # we keep the prompt the same as the one used in the human experiment
    prompt = f"""
            In this task, you will be presented with two sentences. 
            Your job is to select which sentence in a pair is **more** acceptable by **only** returning the index of the sentence: 1 or 2.
            Sentence 1:{sent1}
            Sentence 2:{sent2}
            Which sentence is more acceptable?
    """
    try:
        response = openai.ChatCompletion.create(
            max_tokens=1,
            model=model_name,
            temperature=temp,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract and return the assistant's reply
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def get_accuracy_exp1a(data, temp):
    results = []

    filtered_data = [ex for ex in data if ex["scope"] == "outScope" and (not ex["sent"].startswith("No")) and ex["gender"] != "p"] # only looking at outScope sentences and a vs. every and she/he continuations

    i = 0

    while i < len(filtered_data):
        group = filtered_data[i:i+2]
        accuracy_dict = {}

        good_sent = group[0]["sent"] # first sentence in group is good sentence that starts with "a"
        assert good_sent.startswith("A")
        bad_sent = group[1]["sent"]
        assert bad_sent.startswith("Every")
        
        # randomly swapping the sentence to balance the order effect:
        sents = [good_sent, bad_sent]
        random.shuffle(sents)

        # determine which sentence goes first
        sent1 = sents[0]
        sent2 = sents[1]

        correct_sent = sents[0] if sents[0] == good_sent else sents[1]

        # get the response from the model
        response = get_gpt4o_response(model_name, sent1, sent2, temp)
        
        accuracy_dict["gender"] = group[0]["gender"]
        accuracy_dict["frame"] = group[0]["frame"]

        accuracy_dict["sent1"] = sent1
        accuracy_dict["sent2"] = sent2
        accuracy_dict["raw_response"] = response

        picked_sent = sents[int(response) - 1] # convert to 0-indexed

        if picked_sent == good_sent: # if the model picked the good sentence
            accuracy_dict["correct"] = 1 # correct
        else:
            accuracy_dict["correct"] = 0 # incorrect

        
        results.append(accuracy_dict)
            

        i += 2

    acc = sum([r['correct'] for r in results]) / len(results)
    print(f"Exp1a accuracy is {acc}")

    return results, acc


exp1b_comparisons = {
        "a>if":(5,1),
        "a>whenever":(5,3)
    }

# theoretically, negation_altv is equivalent to DNE
exp2_comparisons = {
    "DNE>negation2_just":(0,4),
    "existential>negation2":(6,2),
    "existential>DNE":(6,0),
    "DNE_infact>negation2_just_infact":(1,5),
    "existential_infact>negation2_infact":(7,3),
}

exp3_comparisons = {
    "disjunction_eitheror>conjunction":(0,3),
    "disjunction_or>conjunction":(2,3),
    "disjunction_eitheror>disjunction_or":(0,2),
    "disjunction_eitheror>disjunction_eitheror_pos":(0,1),
    "disjunction_or>disjunction_eitheror_pos":(2,1)
}

def get_accuracy_exp1b23(rows, comps, group_size, temp):
    results = []

    i = 0

    if dataset == "exp1b":
        rows = [ex for ex in rows if ex["subj"]=="name" and (not ex["sent"].startswith("Every"))]

    progress_bar = tqdm(total=len(rows)/group_size)


    while i < len(rows):
        group = rows[i:i+group_size]
        progress_bar.update(1)
        for comp in comps:
            accuracy_dict = {}

            good_sent = group[comps[comp][0]]["sent"] # good is the first item in the tuple
            bad_sent = group[comps[comp][1]]["sent"] # bad is the second item in the tuple
            # print(good_sent, bad_sent)

            # randomly swapping the sentence to balance the order effect:
            sents = [good_sent, bad_sent]
            random.shuffle(sents)

            # determine which sentence goes first
            sent1 = sents[0]
            sent2 = sents[1]

            # model response
            response = get_gpt4o_response(model_name, sent1, sent2, temp)
            accuracy_dict["raw_response"] = response
            accuracy_dict["sent1"] = sent1
            accuracy_dict["sent2"] = sent2

            accuracy_dict["frame"] = group[0]["frame"]
            accuracy_dict["gender"] = group[0]["gender"]
            accuracy_dict["comp"] = comp

            picked_sent = sents[int(response) - 1] # convert to 0-indexed

            if picked_sent == good_sent: # if the model picked the good sentence
                accuracy_dict["correct"] = 1 # correct
            else:
                accuracy_dict["correct"] = 0 # incorrect

            
            results.append(accuracy_dict)

        i += group_size
    
    accs = []
    for j, comp in enumerate(comps):
        acc = sum([r['correct'] for r in results if r['comp'] == comp]) / len([r['correct'] for r in results if r['comp'] == comp])
        print(f"{comp} accuracy is {acc}")
        accs.append((comp, acc))

    progress_bar.close()
    
    return results, accs


if __name__ == "__main__":

    if dataset == "exp1a":
        res, acc = get_accuracy_exp1a(data, temp=0)
    elif dataset == "exp1b":
        res, acc = get_accuracy_exp1b23(data, exp1b_comparisons, group_size=6, temp=0)
    elif dataset == "exp2":
        res, acc = get_accuracy_exp1b23(data, exp2_comparisons, group_size=8, temp=0)
    elif dataset == "exp3":
        res, acc = get_accuracy_exp1b23(data, exp3_comparisons, group_size=4, temp=0)
    print(f"Accuracies for temp=0: {acc}")
    

    saving_fn = f"results/accuracy/{dataset}_{model_name}.csv"
    print(f"Storing accuracy results to {saving_fn}")
    pd.DataFrame(res).to_csv(saving_fn, index=False)