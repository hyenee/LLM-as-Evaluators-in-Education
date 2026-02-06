# Teacher Feedback Generation & Evaluation System

This repository contains the official implementation code for the paper:

**"Large Language Models as Evaluators in Education: Verification of Feedback Consistency and Accuracy"** *Published in Applied Sciences (MDPI), 2025, 15(2), 671.*

This project implements a pipeline to **generate teacher feedback** for student answers using Large Language Models (LLMs) and **evaluate the consistency and accuracy** of that feedback using an "LLM-as-a-Judge" approach.

## 📝 Reference

If you use this code for your research, please cite the following paper:

> **Large Language Models as Evaluators in Education: Verification of Feedback Consistency and Accuracy** > Hyein Seo, Taewook Hwang, Jeesu Jung, Hyeonseok Kang, Hyuk Namgoong, Yohan Lee, and Sangkeun Jung
> *Applied Sciences*, 2025, 15(2), 671; [https://doi.org/10.3390/app15020671](https://doi.org/10.3390/app15020671)

```bibtex
@article{seo2025llm_evaluators,
  title={Large Language Models as Evaluators in Education: Verification of Feedback Consistency and Accuracy},
  author={Seo, Hyein and Hwang, Taewook and Jung, Jeesu and Kang, Hyeonseok and Namgoong, Hyuk and Lee, Yohan and Jung, Sangkeun},
  journal={Applied Sciences},
  volume={15},
  number={2},
  pages={671},
  year={2025},
  publisher={MDPI},
  doi={10.3390/app15020671}
}

```

---

## 📂 Project Structure

* **`1_generate_feedback.py`**
* Prepares student responses (correct and incorrect) based on the dataset.
* Generates teacher feedback using specified LLMs (e.g., GPT-4o, Claude 3.5 Sonnet, Llama 3.1).
* Supports two generation modes:
* `w_criteria`: Generates feedback guided by specific pedagogical criteria (as defined in the paper).
* `wo_criteria`: Generates feedback without explicit criteria in the prompt.




* **`2_evaluate_feedback.py`**
* Evaluates the generated feedback against 5 specific standards defined in the study (Correct, Revealing, Guidance, Diagnostic, Encouragement).
* Supports two evaluation modes:
* `grouped`: Evaluates all 5 criteria in a single LLM prompt.
* `individual`: Evaluates each criterion separately for more granular analysis.




* **`script.sh`**
* A shell script to automate the entire workflow (Generation  Evaluation) for multiple datasets and models sequentially.



## ⚙️ Prerequisites

To run this project, you need a Python environment with the following dependencies:

* Python 3.x
* PyTorch
* Transformers
* Tqdm
* **Custom `util` package**: The directory must contain a `util` folder with `api_clients.py` and `utils.py` for API handling and file I/O.

## 🚀 Usage

### 1. Automatic Execution

You can run the entire pipeline for various datasets and models using the provided shell script:

```bash
bash script.sh

```

### 2. Manual Execution

#### Step 1: Generate Feedback

Use `1_generate_feedback.py` to create student answers and generate teacher feedback.

```bash
python 1_generate_feedback.py \
    --feedback_gen_mode w_criteria \
    --dataset_name mc160 \
    --model_name gpt-4o \
    --num_responses 5

```

**Arguments:**

* `--feedback_gen_mode`: Mode of generation (`w_criteria` or `wo_criteria`).
* `--dataset_name`: Name of the dataset folder (e.g., `mc160`, `mc500`).
* `--model_name`: The LLM used for generation (e.g., `gpt-4o`, `claude-3-5-sonnet-20240620`, `meta-llama/Meta-Llama-3.1-70B-Instruct`).
* `--num_responses`: Number of feedback candidates to generate per item (Default: 5).

#### Step 2: Evaluate Feedback

Use `2_evaluate_feedback.py` to assess the quality of the generated feedback.

```bash
python 2_evaluate_feedback.py \
    --feedback_gen_mode w_criteria \
    --feedback_eval_mode grouped \
    --dataset_name mc160 \
    --gen_model_name gpt-4o \
    --eval_model_name gpt-4o 
```

**Arguments:**

* `--feedback_gen_mode`: Must match the mode used in generation step.
* `--feedback_eval_mode`: Evaluation method (`grouped` or `individual`).
* `--dataset_name`: Target dataset name.
* `--gen_model_name`: The model name that **generated** the feedback.
* `--eval_model_name`: The model name acting as the **judge** (evaluator).
* `--gpu_id`: Specific GPU IDs to use (e.g., `0,1,2,3`).

## 📊 Evaluation Criteria

The system evaluates feedback based on five binary criteria (Score 1 for satisfied, 0 for not satisfied) as proposed in the paper:

1. **Correct**: The feedback is factually accurate and relevant.
2. **Revealing**: The feedback does **not** directly give away the correct answer.
3. **Guidance**: The feedback provides helpful suggestions to guide the student.
4. **Diagnostic**: The feedback correctly identifies the student's error or misconception.
5. **Encouragement**: The feedback maintains a positive and encouraging tone.

## 📁 Directory Layout

The scripts assume the following directory structure:

```
.
├── data/
│   ├── mc160/              
│   └── mc500/
├── result/
│   └── [dataset_name]/
│       ├── feedback/       
│       └── evaluation/   
├── util/                  
├── 1_generate_feedback.py
├── 2_evaluate_feedback.py
└── script.sh

```