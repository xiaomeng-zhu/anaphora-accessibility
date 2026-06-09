import argparse, json, csv
import pandas as pd

parser = argparse.ArgumentParser(description="Calculate accuracy...")
parser.add_argument("--config", help="Config file name e.g. configs/..")
parser.add_argument("--split", help="def or name for exp1b")
args = parser.parse_args()

config_f = args.config
with open(config_f) as f:
    config_dict = json.load(f)

fn = f"results/raw/{config_dict['dataset']}_{config_dict['model']}.csv"

exp1b_comparisons_def = {

}

exp1b_comparisons_name = {

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

def get_critical_index(row, previous_token):
    tokens = row["tokens"].split("|")
    return tokens.index(previous_token)  # return the index of the first instance!!

def get_log_probs_sum(row):
    return sum([float(f) for f in row["logprobs"][1:-1].split(", ")])

def calculate_accuracy_exp1a(fn, group_size):
    with open(fn) as f:
        rows = [r for r in csv.DictReader(f)]
    i = 0

    accuracies = []

    comps = {
        "every_diff>a_diff":(2,3,0,1),
        "no_diff>a_diff":(4,5,0,1)
    }

    while i < len(rows):
        group = rows[i:i+group_size]
        accuracy_dict = {}
        for comp in comps:
            accuracy_dict = {}

            big_good = get_log_probs_sum(group[comps[comp][0]])
            big_bad = get_log_probs_sum(group[comps[comp][1]])
            small_good = get_log_probs_sum(group[comps[comp][2]])
            small_bad = get_log_probs_sum(group[comps[comp][3]])

            if big_good - big_bad > small_good - small_bad:
                accuracy_dict["correct"] = 1
            else:
                accuracy_dict["correct"] = 0

            accuracy_dict["effect_size"] = (big_good - big_bad) - (small_good - small_bad)

            accuracy_dict["frame"] = group[0]["frame"]
            accuracy_dict["gender"] = group[0]["gender"]
            accuracy_dict["comp"] = comp

            accuracy_dict["good_sent"] = group[comps[comp][3]]["sent"]
            accuracy_dict["bad_sent"] = group[comps[comp][1]]["sent"]
            
            accuracies.append(accuracy_dict)
        
        i += group_size
    
    return accuracies


def calculate_accuracy_exp1b(fn, split, group_size):
    if split == "def":
        new_fn = f"{fn.split('.')[0]}_def.csv"
        group_size = None
    elif split == "name":
        new_fn = f"{fn.split('.')[0]}_name.csv"
        group_size = 6
    else:
        raise ValueError("Undefined split.")
    with open(new_fn) as f:
        rows = [r for r in csv.DictReader(f)]
    
    i = 0
    accuracies = []

    comps = {
        "if_diff>a_diff":(0,1,4,5),
        "whenever_diff>a_diff":(2,3,4,5)
    }
    print(f"group size is {group_size}")

    while i < len(rows):
        group = rows[i:i+group_size]
        accuracy_dict = {}
        for comp in comps:
            accuracy_dict = {}

            big_good = get_log_probs_sum(group[comps[comp][0]])
            big_bad = get_log_probs_sum(group[comps[comp][1]])
            small_good = get_log_probs_sum(group[comps[comp][2]])
            small_bad = get_log_probs_sum(group[comps[comp][3]])

            if big_good - big_bad > small_good - small_bad:
                accuracy_dict["correct"] = 1
            else:
                accuracy_dict["correct"] = 0

            accuracy_dict["effect_size"] = (big_good - big_bad) - (small_good - small_bad)

            accuracy_dict["frame"] = group[0]["frame"]
            accuracy_dict["pred"] = group[0]["pred"]
            accuracy_dict["cont"] = group[0]["cont"]
            accuracy_dict["gender"] = group[0]["gender"]
            accuracy_dict["comp"] = comp

            accuracy_dict["good_sent"] = group[comps[comp][3]]["sent"]
            accuracy_dict["bad_sent"] = group[comps[comp][1]]["sent"]
            
            accuracies.append(accuracy_dict)
        
        i += group_size
    
    return accuracies

def calculate_accuracy_exp23(fn, comps, group_size, measure, previous_token):
    """
    measure = {slor, logprobs}
    """
    
    with open(fn) as f:
        rows = [r for r in csv.DictReader(f)]
    i = 0
    accuracies = []
    while i < len(rows):
        group = rows[i:i+group_size]
        for comp in comps:
            accuracy_dict = {}

            good_row = group[comps[comp][0]] # good is the first item in the tuple
            bad_row = group[comps[comp][1]] # bad is the second item in the tuple
            # print(good_row["sentence_type"], bad_row["sentence_type"])

            if measure == "slor":
                slor_good = float(good_row["slor_nofirst"])
                slor_bad = float(bad_row["slor_nofirst"])

                if slor_good > slor_bad:
                    accuracy_dict["correct"] = 1
                else:
                    accuracy_dict["correct"] = 0

                accuracy_dict["effect_size"] = slor_good - slor_bad
            elif measure == "logprobs":
                critical_index_good = get_critical_index(good_row, previous_token)
                critical_index_bad = get_critical_index(bad_row, previous_token)

                continuation_sum_good = sum([float(f) for f in good_row["logprobs"][1:-1].split(", ")][critical_index_good:])
                continuation_sum_bad = sum([float(f) for f in bad_row["logprobs"][1:-1].split(", ")][critical_index_bad:])

                if continuation_sum_good > continuation_sum_bad:
                    accuracy_dict["correct"] = 1
                else:
                    accuracy_dict["correct"] = 0

                accuracy_dict["effect_size"] = continuation_sum_good - continuation_sum_bad

            accuracy_dict["frame"] = group[0]["frame"]
            accuracy_dict["gender"] = group[0]["gender"]
            accuracy_dict["comp"] = comp
            if "a_in>a_out" in comps: # which means exp1a:
                pass
            else:
                accuracy_dict["subject"] = group[0]["subject"]

            accuracy_dict["good_sent"] = good_row["sent"]
            accuracy_dict["bad_sent"] = bad_row["sent"]

            accuracies.append(accuracy_dict)
            # print(comp, slor_good, slor_bad, accuracy_dict)
        i += group_size
    
    return accuracies

out_fn = f"results/accuracy/{config_dict['dataset']}_{config_dict['model']}_{args.split}.csv"
print(f"Storing accuracy results to {out_fn}")

if config_dict['dataset'] == "exp1a":
    pd.DataFrame(calculate_accuracy_exp1a(fn, 6)).to_csv(out_fn, index=False)
elif config_dict['dataset'] == "exp1b":
    pd.DataFrame(calculate_accuracy_exp1b(fn, args.split, 8)).to_csv(out_fn, index=False)
elif config_dict['dataset'] == "exp2":
    pd.DataFrame(calculate_accuracy_exp23(fn, exp2_comparisons, 8 , "logprobs", ".")).to_csv(out_fn, index=False)
elif config_dict['dataset'] == "exp3":
    pd.DataFrame(calculate_accuracy_exp23(fn, exp3_comparisons, 4 , "slor", None)).to_csv(out_fn, index=False)