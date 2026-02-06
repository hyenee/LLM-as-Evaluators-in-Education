echo "Running script - Generate Feedback (GenMode=w_criteria, Dataset=MC160)..."
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc160 --model_name gpt-4o
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc160 --model_name claude-3-5-sonnet-20240620
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc160 --model_name meta-llama/Meta-Llama-3.1-70B-Instruct

echo "Running script - Generate Feedback (GenMode=w_criteria, Dataset=MC500)..."
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc500 --model_name gpt-4o
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc500 --model_name claude-3-5-sonnet-20240620
python 1_generate_feedback.py --feedback_gen_mode w_criteria --dataset_name mc500 --model_name meta-llama/Meta-Llama-3.1-70B-Instruct


echo "Running script - Generate Feedback (GenMode=wo_criteria, Dataset=MC160)..."
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc160 --model_name gpt-4o
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc160 --model_name claude-3-5-sonnet-20240620
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc160 --model_name meta-llama/Meta-Llama-3.1-70B-Instruct

echo "Running script - Generate Feedback (GenMode=wo_criteria, Dataset=MC500)..."
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc500 --model_name gpt-4o
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc500 --model_name claude-3-5-sonnet-20240620
python 1_generate_feedback.py --feedback_gen_mode wo_criteria --dataset_name mc500 --model_name meta-llama/Meta-Llama-3.1-70B-Instruct


echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC160, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC160, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC160, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC500, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC500, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Grouped, Dataset=MC500, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it


echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC160, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC160, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC160, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC500, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC500, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=w_criteria, EvalMode=Individual, Dataset=MC500, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it


echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC160, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode w_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC160, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC160, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC500, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC500, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Grouped, Dataset=MC500, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode grouped --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it


echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC160, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC160, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC160, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc160 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC500, Model=GPT-4o)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name gpt-4o --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC500, Model=Claude-3.5-Sonnet)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name claude-3-5-sonnet-20240620 --eval_model_name google/gemma-2-27b-it

echo "Running script - Evaluate Feedback (GenMode=wo_criteria, EvalMode=Individual, Dataset=MC500, Model=Llama-3.1-70B-Instruct)..."
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name gpt-4o 
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-9b-it
python 2_evaluate_feedback.py --feedback_gen_mode wo_criteria -feedback_eval_mode individual --dataset_name mc500 --gen_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --eval_model_name google/gemma-2-27b-it

echo "Script execution completed."