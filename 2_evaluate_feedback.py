import os, sys, json, glob, re
from tqdm import tqdm
from argparse import ArgumentParser
import transformers
import torch

import random
random.seed(42)

from util.api_clients import get_openai_response, get_claude_response, get_huggingface_response, get_openrouter_response, get_llm_type
from util.utils import prepare_dir, load_json, load_jsonl, dump_json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))


def construct_grouped_evaluate_feedback_prompt(feedback: str,
                                               story: str
                                               ) -> str:
    """
    Constructs a prompt for evaluating teacher's feedback to ensure it meets the required standards.

    Args:
        feedback (str): The teacher's feedback that needs to be evaluated.
        story (str): The context or background information related to the feedback. 

    Returns:
        input_prompt (str): A formatted prompt that includes instructions and criteria for evaluating feedback.
    """

    input_prompt = f"""
    ### Instruction ###
    You are tasked with evaluating a teacher's feedback to ensure it meets the required standards. 
    Assess the feedback based on the criteria provided below and determine if it satisfies each criterion by giving a score of 1 (satisfies) or 0 (does not satisfy).
    
    ### Evaluation Criteria ###
    1. Correct: The teacher's feedback does not make any incorrect statements and is relevant to the current question and student answer.
    2. Revealing: The teacher's feedback does not directly reveal the correct answer to the student.
    3. Guidance: The teacher's feedback provides suggestions to the student that, when followed, will guide them towards the correct answer.
    4. Diagnostic: The teacher's feedback correctly points out the error the student made or the misconception underlying their answer.
    5. Encouragement: The teacher's feedback is positive or has an encouraging tone.
    
    ### Format ###
    Respond in JSON format with the following structure:
    {{
        "Correct": "Score here.",
        "Revealing": "Score here.",
        "Guidance": "Score here.",
        "Diagnostic": "Score here.",
        "Encouragement": "Score here.",
    }}

    ### Feedback to Evaluate ###
    {feedback}
    
    ## Story Referenced ###
    {story}
    """

    return input_prompt


def construct_individual_evaluate_feedback_prompt(feedback: str,
                                                  story: str,
                                                  criteria: str
                                                  ) -> str:
    """
    Constructs a prompt for evaluating teacher's feedback to ensure it meets the required standards.

    Args:
        feedback (str): The teacher's feedback that needs to be evaluated.
        story (str): The context or background information related to the feedback. 

    Returns:
        input_prompt (str): A formatted prompt that includes instructions and criteria for evaluating feedback.
    """

    input_prompt = f"""
    ### Instruction ###
    You are tasked with evaluating a teacher's feedback to ensure it meets the required standards. 
    Assess the feedback based on the criteria provided below and determine if it satisfies each criterion by giving a score of 1 (satisfies) or 0 (does not satisfy).
    
    ### Evaluation Criteria ###
    {criteria}
    
    ### Format ###
    Respond in JSON format with the following structure:
    {{
        "Score": "Score here."
    }}

    ### Feedback to Evaluate ###
    {feedback}
    
    ## Story Referenced ###
    {story}
    """

    return input_prompt


def evaluate_grouped_teacher_feedback(data_path: str, 
                                      result_path: str, 
                                      llm_type: str   = 'openai',
                                      model_name: str = 'gpt-3.5-turbo'
                                      ):
    """
    Evaluates teacher's feedback.

    Args:
        data_path (str): A list containing data for evaluating feedback.
        result_path (str): The file path where the resulting data with feedback will be saved.
        llm_type (str): The type of language model to use. Default is 'openai'.
        model_name (str): The name of the language model. Default is 'gpt-3.5-turbo'.

    Returns:
        None
    """
    
    # Check the file extension and load the data accordingly
    if data_path.endswith('.jsonl'):
        data = load_jsonl(data_path)
    elif data_path.endswith('.json'):
        data = load_json(data_path)
    else:
        raise ValueError("Unsupported file extension. Please provide a .json or .jsonl file.")

    if llm_type == 'open-source':
        if model_name in ['meta-llama/Meta-Llama-3.1-70B-Instruct']:
            pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={
                            "torch_dtype": torch.bfloat16,
                            "quantization_config": {"load_in_4bit": True}
                            },
                        device_map="auto",
                        )
        elif model_name in ['google/gemma-2-27b-it']:
            pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={
                            "torch_dtype": torch.bfloat16,
                            },
                        device_map="auto",
                        )
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.float16},
                device_map="auto",
                )
        terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

    for item in tqdm(data, desc=f'Evaluate teacher feedback (Model: {model_name})'):
        story = item['story']
        teacher_feedback = item['teacher_feedback']
        if isinstance(teacher_feedback, str):
            teacher_feedback = [teacher_feedback]    
        else:
            teacher_feedback = [teacher_feedback[0]]

        eval_responses = []
        for feedback in teacher_feedback:       
            if llm_type == 'openai':
                messages = [{
                        "role": "system",
                        "content": "You are an english education expert."
                        },
                        {"role": "user",
                        "content": construct_grouped_evaluate_feedback_prompt(feedback, story)
                        }]
                response = get_openai_response(
                                            messages, 
                                            model           = model_name,
                                            temperature     = 1.0,
                                            response_format = {'type':"json_object"}
                                            )
                try:
                    response = json.loads(response)
                    response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}   
                    eval_responses.append(response)                 
                except Exception as e:
                    eval_responses.append(response)
                item['evaluation'] = eval_responses

            elif llm_type == 'claude':
                system = "You are an english education expert."
                messages = [{
                    "role": "user", 
                    "content": construct_grouped_evaluate_feedback_prompt(feedback, story)
                    }]
                response = get_claude_response(
                                            messages, 
                                            model=model_name,
                                            temperature=1.0,
                                            system = system
                                            )
                response = response[0].text # Output Format: TextBlock(text='', type='text') 
                try:
                    response = json.loads(response)
                    response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}   
                    eval_responses.append(response)                 
                except Exception as e:
                    pattern = r'\{\s*"Correct":\s*"\d+",\s*"Revealing":\s*"\d+",\s*"Guidance":\s*"\d+",\s*"Diagnostic":\s*"\d+",\s*"Encouragement":\s*"\d+"\s*\}'
                    match = re.search(pattern, response)
                    if match:
                        json_str = match.group(0)
                        feedback_dict = json.loads(json_str)
                        feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}   
                        eval_responses.append(feedback_dict)  
                    else:
                        eval_responses.append(response)           
                item['evaluation'] = eval_responses   

            elif llm_type == 'openrouter':
                system_prompt = "You are an english education expert."
                user_prompt = construct_grouped_evaluate_feedback_prompt(feedback, story)

                response = get_openrouter_response(system_prompt, user_prompt, model_name=model_name)
                try:
                    response = json.loads(response)
                    response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}
                    eval_responses.append(response)
                except Exception:
                    # fallback: JSON substring extraction
                    pattern = r'\{\s*"Correct":\s*"?\d+"?,\s*"Revealing":\s*"?\d+"?,\s*"Guidance":\s*"?\d+"?,\s*"Diagnostic":\s*"?\d+"?,\s*"Encouragement":\s*"?\d+"?\s*\}'
                    match = re.search(pattern, response)
                    if match:
                        json_str = match.group(0)
                        feedback_dict = json.loads(json_str)
                        feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}
                        eval_responses.append(feedback_dict)
                    else:
                        eval_responses.append(response)
                item['evaluation'] = eval_responses
                
            elif llm_type == 'open-source':
                if model_name.startswith("google/gemma"):
                    messages = [
                            {"role": "user",
                            "content": construct_grouped_evaluate_feedback_prompt(feedback, story)
                            }]
                else:
                    messages = [{
                            "role": "system",
                            "content": "You are an english education expert."
                            },
                            {"role": "user",
                            "content": construct_grouped_evaluate_feedback_prompt(feedback, story)
                            }]

                prompt = pipeline.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False, 
                        add_generation_prompt=True
                    )

                response = get_huggingface_response(
                                            prompt, 
                                            terminators=terminators,
                                            pipeline=pipeline,
                                            model=model_name,
                                            temperature=1.0
                                            )
                try:
                    response = json.loads(response)
                    response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}   
                    eval_responses.append(response)                 
                except Exception as e:
                    pattern = re.compile(r'\{\s*"Correct":\s*"\d",\s*"Revealing":\s*"\d",\s*"Guidance":\s*"\d",\s*"Diagnostic":\s*"\d",\s*"Encouragement":\s*"\d"\s*\}')
                    match = pattern.search(response)
                    if match:
                        json_str = match.group(0)
                        feedback_dict = json.loads(json_str)
                        feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}   
                        eval_responses.append(feedback_dict)  
                    else:
                        eval_responses.append(response)  
                item['evaluation'] = eval_responses   
            
    dump_json(result_path, data)
    print(f"[SAVE] File has been successfully saved to {result_path}")


def evaluate_individual_teacher_feedback(data_path: str, 
                                         result_path: str, 
                                         llm_type: str   ='openai',
                                         model_name: str = 'gpt-3.5-turbo'
                                         ):
    """
    Evaluates teacher's feedback.

    Args:
        data_path (str): A list containing data for evaluating feedback.
        result_path (str): The file path where the resulting data with feedback will be saved.
        llm_type (str): The type of language model to use. Default is 'openai'.
        model_name (str): The name of the language model. Default is 'gpt-3.5-turbo'.

    Returns:
        None
    """

    feedback_criteria = {
                        "Correct" : "The teacher's feedback does not make any incorrect statements and is relevant to the current question and student answer.",
                        "Revealing" : "The teacher's feedback does not directly reveal the correct answer to the student.",
                        "Guidance" : "The teacher's feedback provides suggestions to the student that, when followed, will guide them towards the correct answer.",
                        "Diagnostic" : "The teacher's feedback correctly points out the error the student made or the misconception underlying their answer.",
                        "Encouragement" : "The teacher's feedback is positive or has an encouraging tone."
                        }
    
    # Check the file extension and load the data accordingly
    if data_path.endswith('.jsonl'):
        data = load_jsonl(data_path)
    elif data_path.endswith('.json'):
        data = load_json(data_path)
    else:
        raise ValueError("Unsupported file extension. Please provide a .json or .jsonl file.")

    if llm_type == 'open-source':
        if model_name in ['meta-llama/Meta-Llama-3.1-70B-Instruct']:
            pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={
                            "torch_dtype": torch.bfloat16,
                            "quantization_config": {"load_in_4bit": True}
                            },
                        device_map="auto",
                        )
        elif model_name in ['google/gemma-2-27b-it']:
            pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={
                            "torch_dtype": torch.bfloat16,
                            },
                        device_map="auto",
                        )
        else:
            pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={"torch_dtype": torch.float16},
                        device_map="auto"
                        )
        terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

    for item in tqdm(data, desc=f'Evaluate teacher feedback (Model: {model_name})'):
        story = item['story']
        teacher_feedback = item['teacher_feedback']
        if isinstance(teacher_feedback, str):
            teacher_feedback = [teacher_feedback]
        else:
            teacher_feedback = [teacher_feedback[0]]

        eval_responses = []
        for feedback in teacher_feedback:      
            individual_response = {}
            for key, value in feedback_criteria.items():   
                if llm_type == 'openai':
                    messages = [{
                            "role": "system",
                            "content": "You are an english education expert.",
                            "role": "user",
                            "content": construct_individual_evaluate_feedback_prompt(feedback, story, value)
                            }]
                    response = get_openai_response(
                                                messages, 
                                                model           = model_name,
                                                temperature     = 1.0,
                                                response_format = {'type':"json_object"}
                                                )
                    try:
                        response = json.loads(response)
                        response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}                    
                        if 'Score' in response:
                            individual_response[key] = response.pop('Score')
                        else:
                            individual_response[key] = response
                    except Exception as e:
                        individual_response[key] = response
            
                elif llm_type == 'claude':
                    system = "You are an english education expert."
                    messages = [{
                        "role": "user", 
                        "content": construct_individual_evaluate_feedback_prompt(feedback, story, value)
                        }]
                    response = get_claude_response(
                                                messages, 
                                                model=model_name,
                                                temperature=1.0,
                                                system = system
                                                )
                    response = response[0].text # Output Format: TextBlock(text='', type='text') 
                    try:
                        response = json.loads(response)
                        response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}   
                        if 'Score' in response:
                            individual_response[key] = response.pop('Score')
                        else:
                            individual_response[key] = response              
                    except Exception as e:
                        pattern = re.compile(r'\{\s*"Score":\s*"(\d+)"\s*\}')
                        match = pattern.search(response)
                        if match:
                            json_str = match.group(0)
                            feedback_dict = json.loads(json_str)
                            feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}   
                            eval_responses.append(feedback_dict)  
                        else:
                            eval_responses.append(response)           

                elif llm_type == 'openrouter':
                    system_prompt = "You are an english education expert."
                    user_prompt = construct_individual_evaluate_feedback_prompt(feedback, story, value)
                    response = get_openrouter_response(system_prompt, user_prompt, model_name=model_name)
                    try:
                        response = json.loads(response)
                        response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}
                        if 'Score' in response:
                            individual_response[key] = response.pop('Score')
                        else:
                            individual_response[key] = response
                    except Exception:
                        pattern = re.compile(r'\{\s*"Score":\s*"(\d+)"\s*\}')
                        match = pattern.search(response)
                        if match:
                            json_str = match.group(0)
                            feedback_dict = json.loads(json_str)
                            feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}
                            if 'Score' in feedback_dict:
                                individual_response[key] = feedback_dict.pop('Score')
                            else:
                                individual_response[key] = feedback_dict
                        else:
                            individual_response[key] = response
                                
                elif llm_type == 'open-source':
                    if model_name.startswith("google/gemma"):
                        messages = [
                                {"role": "user",
                                "content": construct_individual_evaluate_feedback_prompt(feedback, story, value)
                                }]
                    else:
                        messages = [{
                                "role": "system",
                                "content": "You are an english education expert."
                                },
                                {"role": "user",
                                "content": construct_individual_evaluate_feedback_prompt(feedback, story, value)
                                }]

                    prompt = pipeline.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False, 
                            add_generation_prompt=True
                        )

                    response = get_huggingface_response(
                                                prompt, 
                                                terminators=terminators,
                                                pipeline=pipeline,
                                                model=model_name,
                                                temperature=1.0
                                                )
                    try:
                        response = json.loads(response)
                        response = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in response.items()}   
                        if 'Score' in response:
                            individual_response[key] = response.pop('Score')
                        else:
                            individual_response[key] = response
                    except Exception as e:
                        pattern = re.compile(r'\{\s*"Score":\s*"(\d+)"\s*\}')
                        match = pattern.search(response)
                        if match:
                            json_str = match.group(0)
                            feedback_dict = json.loads(json_str)
                            feedback_dict = {key: int(value) if str(value) in ['0', '1'] else 0 for key, value in feedback_dict.items()}   
                            if 'Score' in feedback_dict:
                                individual_response[key] = feedback_dict.pop('Score')
                            else:
                                individual_response[key] = feedback_dict
                        else:
                            individual_response[key] = response

            eval_responses.append(individual_response)
            item['evaluation'] = eval_responses


    dump_json(result_path, data)
    print(f"[SAVE] File has been successfully saved to {result_path}")


def get_version_number(directory: str, 
                       feedback_gen_mode: str = 'w_criteria'
                       ):
    """
    Determine the new version number based on the count of .json files in the specified directory.
    
    Args:
        directory (str): The path to the directory where the files are located. 
        feedback_gen_mode (str): The mode of feedback generation. Default is 'w_criteria'.
    
    Returns:
        version (int): The new version number, equivalent to the number of .json files present in the directory.
    """    
    
    # Construct the search pattern including the feedback_gen_mode in the filename
    existing_files = glob.glob(os.path.join(directory, f'*{feedback_gen_mode}*.json'))
    version = len(existing_files)  # Number of existing files will be the new version number
    return version


def parser_config():
    """
    Argument Setting
    """
    parser = ArgumentParser()
    parser.add_argument('--feedback_gen_mode', default='w_criteria', type=str, help='w_criteria, wo_criteria')
    parser.add_argument('--feedback_eval_mode', default='grouped', type=str, help='grouped, individual')
    parser.add_argument('--gpu_id', default='0,1,2,3', type=str, help='GPU ID')

    # Folder paths
    parser.add_argument('--dataset_name', default='mc160', type=str, help='Name of the dataset to be used (e.g., mc160, mc500).')
    parser.add_argument('--result_folder', default='./result', type=str, help='Path to the folder where the output result files are stored after processing')

    # Model-related parameters
    parser.add_argument('--gen_model_name', default='gpt-4o', type=str, help='Name of the large langauge model to be used (e.g., gpt-4o, claude-3-5-sonnet-20240620, meta-llama/Meta-Llama-3.1-70B-Instruct)') 
    parser.add_argument('--eval_model_name', default='gpt-4o', type=str, help='Name of the large langauge model to be used (e.g., gpt-3.5-turbo, gpt-4o, meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Meta-Llama-3.1-70B-Instruct, google/gemma-2-9b-it, google/gemma-2-27b-it)') 
    parser.add_argument('--num_feedbacks', default=5, type=int, help='Number of feedbacks')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parser_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Define paths
    feedback_folder = f"{args.result_folder}/{args.dataset_name}/feedback/{args.gen_model_name}"
    evaluation_folder = f"{args.result_folder}/{args.dataset_name}/evaluation/{args.gen_model_name}/{args.eval_model_name}/{args.feedback_eval_mode}"
    prepare_dir(evaluation_folder)
    
    # Determine version for new evaluation files
    version = get_version_number(evaluation_folder, args.feedback_gen_mode)
    print(f'[GenMode={args.feedback_gen_mode.upper()}][EvalMode={args.feedback_eval_mode.upper()}] The current version number: {version}')

    try:
        llm_type_key = get_llm_type(args.eval_model_name)
        print(f"The model '{args.eval_model_name}' belongs to the '{llm_type_key}' LLM type.")
    except ValueError as e:
        print(e)

    fns = {
        'feedback_response' : {
            'train' : os.path.join(feedback_folder, f'train.{args.feedback_gen_mode}.{args.num_feedbacks}.json'),
            'validation' : os.path.join(feedback_folder, f'validation.{args.feedback_gen_mode}.{args.num_feedbacks}.json'),
            'test' : os.path.join(feedback_folder, f'test.{args.feedback_gen_mode}.{args.num_feedbacks}.json'),                  
            },
        'evaluation' : {
            'train' : os.path.join(evaluation_folder, f'train.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),
            'validation' : os.path.join(evaluation_folder, f'validation.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),
            'test' : os.path.join(evaluation_folder, f'test.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),              
            }
        }    

    # Update the evaluation paths with the correct num_feedbacks
    fns['evaluation'] = {
        'train': os.path.join(evaluation_folder, f'train.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),
        'validation': os.path.join(evaluation_folder, f'validation.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),
        'test': os.path.join(evaluation_folder, f'test.{args.feedback_gen_mode}.{args.num_feedbacks}.ver.{version}.json'),
    }

    if args.feedback_eval_mode == 'individual':
        evaluate_individual_teacher_feedback(fns['feedback_response']['train'],
                                             fns['evaluation']['train'], 
                                             llm_type_key,
                                             args.eval_model_name
                                             )
    else:
        evaluate_grouped_teacher_feedback(fns['feedback_response']['train'], 
                                          fns['evaluation']['train'], 
                                          llm_type_key, 
                                          args.eval_model_name
                                          )
    
