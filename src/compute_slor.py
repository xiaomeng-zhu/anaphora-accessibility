import csv
import json
import pandas as pd
import ast
import os
import math


# Load Freq_Dic according to Model/Tokenizer Type
def load_freq_dic(tokenzier):
    with open(f'utils/freq_dics/{tokenzier}_OpenWebText1.json', 'r') as file1:
        dic1 = json.load(file1)
        
    with open(f'utils/freq_dics/{tokenzier}_OpenWebText2.json', 'r') as file2:
        dic2 = json.load(file2)
        
    dic = {key: dic1.get(key, 0) + dic2.get(key, 0) for key in set(dic1) | set(dic2)}
    TOTAL = sum(dic.values())
    
    return dic, TOTAL

def get_log_prob(token, dic, TOTAL):
    return math.log(dic[token] / TOTAL)

# Function for computing SLOR score for each row in the DataFrame
def compute_slor(row, dic, TOTAL):
    tokens = row['tokens'].split('|')
    #tokens = [token.replace(' ', 'Ġ') for token in tokens]
    unigram_prob = sum([get_log_prob(token, dic, TOTAL) for token in tokens])
    model_prob = sum(ast.literal_eval(row['logprobs']))
    slor_score = (model_prob - unigram_prob) / len(tokens)
    return slor_score

def compute_slor_noFirst(row, dic, TOTAL):
    tokens = row['tokens'].split('|')
    #tokens = [token.replace(' ', 'Ġ') for token in tokens]
    unigram_prob = sum([get_log_prob(token, dic, TOTAL) for token in tokens[1:]])
    model_prob = sum(ast.literal_eval(row['logprobs']))
    slor_score = (model_prob - unigram_prob) / len(tokens[1:])
    return slor_score


# Run the script to compute SLOR scores and add them as a new column in the CSV files
if __name__ == "__main__":

    all_files = [
        "results/raw/exp3_babbage.csv",
        "results/raw/exp3_davinci.csv",
        "results/raw/exp3_Llama3-1-8B-Instruct.csv",
        "results/raw/exp3_Llama3-1-8B.csv",
        "results/raw/exp3_Llama3-2-1B.csv",
        "results/raw/exp3_Llama3-2-3B.csv"
    ]

    for file_name in all_files:
        print(f'Processing {file_name}...')
        
        # file_path = f'{directory_path}/{file_name}'
        file_path = f'{file_name}'
        tokenizer = 'Llama3' if 'Llama3' in file_path else 'GPT3'
        dic, TOTAL = load_freq_dic(tokenizer)
        new_column_name = 'slor_nofirst'
        temp_file_path = file_path + '.tmp'

        # Open the original file to read and a temporary file to write
        with open(file_path, 'r', encoding='utf-8') as infile, open(temp_file_path, 'w', encoding='utf-8', newline='') as outfile:
            # Read the original file and write to the temp file with modifications
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + [new_column_name]  # Add the new column name
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row[new_column_name] = compute_slor_noFirst(row, dic, TOTAL)  # Add the computed value to the new column
                # print(compute_slor(row, dic, TOTAL))
                # print(compute_slor_noFirst(row, dic, TOTAL))
                # break
                writer.writerow(row)

        # Replace the original file with the modified temporary file
        os.replace(temp_file_path, file_path)
