import json

if __name__ == "__main__":
    MODEL_IDS = [
        'Llama3-2-1B',
        'Llama3-2-3B',
        'Llama3-1-8B',
        'Llama3-1-8B-Instruct',
        'davinci',
        'babbage'

    ]

    MODEL_NAME_CARDS = [
        'meta-llama/Llama-3.2-1B',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.1-8B',
        'meta-llama/Llama-3.1-8B-Instruct',
        'davinci-002',
        'babbage-002'
    ]

    DATASETS = [
        "exp1a", "exp1b", "exp2", 'exp3'
    ]

    for model_id, model_name_hf in zip(MODEL_IDS, MODEL_NAME_CARDS):
        for d in DATASETS:
            config_d = {
                "dataset": d,
                "model": model_id,
                "model_hf": model_name_hf
            }
            with open(f"configs/{d}_{model_id}.json", "w") as f:
                f.write(json.dumps(config_d))