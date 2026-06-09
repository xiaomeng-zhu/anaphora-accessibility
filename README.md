# Anaphora Accessibility
Repository for the ACL 2025 paper "Meaning Beyond Truth Conditions: Evaluating Discourse Level
Understanding via Anaphora Accessibility"


## Repository Structure

```text
anaphora-accessibility/
├── configs/
│   └── ...                         # Experiment configuration files
│
├── dataset/
│   ├── templates/
│   │   └── ...                     # Sentence templates
│   ├── exp1a_metadata.jsonl         # Dataset for Experiment 1a
│   ├── exp1b_metadata.jsonl         # Dataset for Experiment 1b
│   ├── exp2_metadata.jsonl          # Dataset for Experiment 2
│   └── exp3_metadata.jsonl          # Dataset for Experiment 3
│
├── plots/
│   └── ...                          # Generated figures and visualizations
│
├── results/
│   ├── accuracy/
│   │   └── ...                      # Processed evaluation results
│   └── raw/
│       └── ...                      # Raw model responses
│
├── src/
│   ├── analysis.R                   # Statistical analysis and plotting
│   ├── calculate_accuracy.py        # Calculate accuracy from model outputs
│   ├── compute_slor.py              # Compute SLOR-based metrics
│   ├── generate_config.py           # Generate experiment configurations
│   ├── generate_metadata.py         # Generate experiment datasets
│   ├── gpt_family.py                # Inference for GPT-family models
│   ├── llama_family.py              # Inference for Llama-family models
│   └── prompt.py                    # Prompt construction for GPT-4o
│
├── utils/
│   └── freq_dics/
│       └── ...                      # Frequency dictionaries used for SLOR
│
└── README.md
```


## Configuration

Experiment-specific configuration files are stored in `configs/`, which are json files storing dataset and model combinations.


## Dataset

We release the dataset files for the following experiments:

| Experiment    | Dataset file                  | Description                                    |
| ------------- | ------------------------------ | ---------------------------------------------- |
| Experiment 1a | `dataset/exp1a_metadata.jsonl` | Existential vs. Universal (*Every*) |
| Experiment 1b | `dataset/exp1b_metadata.jsonl` | Existential vs. Universal (*Donkey conditionals*) |
| Experiment 2  | `dataset/exp2_metadata.jsonl`  | Negation  |
| Experiment 3  | `dataset/exp3_metadata.jsonl`  | Disjunction  |


Each line in a `.jsonl` file contains one JSON object representing an experimental item.


## Running the Pipeline

An example of running llama family models on our dataset using data frame experiment 1a:

```bash
python src/llama_family.py --config configs/exp1a_Llama3-1-8B.json
```


Use `src/calculate_accuracy.py` to calculate accuracy on our dataset from raw model logits.

Example:

```bash
python src/calculate_accuracy.py --config configs/exp1a_Llama3-1-8B.json
```

Use the R script `src/analysis.R` to perform statistical analysis and generate visualizations:


## Citation


```bibtex
@inproceedings{zhu-etal-2025-meaning,
    title = "Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility",
    author = "Zhu, Xiaomeng  and
      Zhou, Zhenghao  and
      Charlow, Simon  and
      Frank, Robert",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.432/",
    doi = "10.18653/v1/2025.acl-long.432",
    pages = "8824--8842",
    ISBN = "979-8-89176-251-0",
    abstract = "We present a hierarchy of natural language understanding abilities and argue for the importance of moving beyond assessments of understanding at the lexical and sentence levels to the discourse level. We propose the task of anaphora accessibility as a diagnostic for assessing discourse understanding, and to this end, present an evaluation dataset inspired by theoretical research in dynamic semantics. We evaluate human and LLM performance on our dataset and find that LLMs and humans align on some tasks and diverge on others. Such divergence can be explained by LLMs' reliance on specific lexical items during language comprehension, in contrast to human sensitivity to structural abstractions."
}
```

## Contact

For questions, please contact [Miranda Zhu](xiaomeng-zhu.github.io) or [Herbert Zhou](https://herbert-zhou.github.io/).
