import os, sys, json, re
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from tqdm import tqdm
from argparse import ArgumentParser
import transformers
import torch
import random
random.seed(42)

from util.api_clients import get_openai_response, get_claude_response, get_huggingface_response, get_llm_type
from util.utils import prepare_dir, load_json, load_jsonl, dump_json, dump_jsonl

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))


def construct_teacher_feedback_prompt(story: str,
                                      question: str,
                                      student_incorrect_response: str,
                                      student_correct_response: str,
                                      feedback_gen_mode = 'w_criteria'
                                      ) -> str:
    """
    Constructs a prompt for generating teacher's feedback to student response.

    Args:
        story (str): The context or background information related to the question. 
        question (str): The specific question posed to the student, which the feedback will address.
        student_incorrect_response (str): The student's incorrect response to a question or task.
        student_correct_response (str): The correct response for the question or task.
        feedback_gen_mode (str): The mode or style of feedback to generate. Default is 'w_criteria'.

    Returns:
        input_prompt (str): A formatted prompt that includes instructions and criteria for generating feedback.
    """

    if feedback_gen_mode == 'w_criteria':
        input_prompt = f"""
        ### Instruction ###
        You are an English teacher tasked with providing feedback to students. 
        Your goal is to help the student understand their mistake and guide them toward the correct answer without directly providing it.
        The feedback must be limited to one sentence.
        
        ### Feedback Criteria ###
        1. The teacher's feedback does not make any incorrect statements and is relevant to the current question and student answer.
        2. The teacher's feedback does not directly reveal the correct answer to the student.
        3. The teacher's feedback provides suggestions to the student that, when followed, will guide them towards the correct answer.
        4. The teacher's feedback correctly points out the error the student made or the misconception underlying their answer.
        5. The teacher's feedback is positive and has an encouraging tone.
        
        ### Format ###
        Respond in JSON format with the following structure:
        {{
            "feedback": "Your feedback here."
        }}

        ### Story ###
        {story}

        ### Question ###
        {question}

        ### Incorrect Response ###
        {student_incorrect_response}

        ### Correct Response ###
        {student_correct_response}
        """

    elif feedback_gen_mode == 'wo_criteria':
        input_prompt = f"""
        ### Instruction ###
        You are an English teacher tasked with providing feedback to students. 
        Your goal is to provide feedback that guides the student from an incorrect answer to the correct one.
        The feedback must be limited to one sentence.

        ### Format ###
        Respond in JSON format with the following structure:
        {{
            "feedback": "Your feedback here."
        }}

        ### Story ###
        {story}

        ### Question ###
        {question}

        ### Incorrect Response ###
        {student_incorrect_response}

        ### Correct Response ###
        {student_correct_response}
        """

    return input_prompt


def generate_student_answer(data_path: str, 
                            result_path: str
                            ):
    """
    Generates the student's correct answer and incorrect answer based on the given answer options and the correct answer,
    and saves the results in a JSONL file.
    
    Args:
        data_path (str): Path to the input JSON or JSONL file.
        result_path (str): Path to the output JSONL file.

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

    # Prepare the output data list
    output_data = []

    for item in tqdm(data, desc=f'Generate student answer'):
        answer_options = item['answer_options']
        answer = item['answer']
        
        correct_option = answer_options.get(answer)
        if correct_option is None:
            raise ValueError(f"Invalid answer key '{answer}' provided.")

        # Create a list of incorrect options
        incorrect_options = [key for key in answer_options if key != answer]
        
        # Select a random incorrect answer
        random_incorrect_answer_key = random.choice(incorrect_options)
        random_incorrect_answer = answer_options[random_incorrect_answer_key]
        
        # Generate the formatted correct answer
        student_correct_response = f"The answer is {answer}: {correct_option}."
        
        # Generate the formatted random incorrect answer
        student_incorrect_response = f"The answer is {random_incorrect_answer_key}: {random_incorrect_answer}."

        # Add the answers to the item
        item['student_correct_response'] = student_correct_response
        item['student_incorrect_response'] = student_incorrect_response
        
        # Add the modified item to the output data list
        output_data.append(item)

    # Save the output data to a JSONL file
    dump_jsonl(result_path, output_data)
    print(f"[SAVE] File has been successfully saved to {result_path}")


def generate_teacher_feedback(data_path: str, 
                              result_path: str,
                              llm_type: str      = 'openai',
                              model_name: str    = 'gpt-3.5-turbo',
                              feedback_gen_mode: str = 'w_criteria',
                              num_responses: int     = 1
                              ):
    """
    Generate teacher's feedbacks for student responses.

    Args:
        data_path (str): The file path to the JSON or JSONL file containing data for generating feedback.
        result_path (str): The file path where the resulting data with generated feedback will be saved.
        llm_type (str): The type of language model to use. Default is 'openai'.
        model_name (str): The name of the language model. Default is 'gpt-3.5-turbo'.
        feedback_gen_mode (str): Specifies the mode of feedback generation. Default is 'w_criteria'.
        num_responses (int): The number of feedback responses to generate per data item. Default is 1.

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
                            # "quantization_config": {"load_in_4bit": True}
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

    for item in tqdm(data, desc=f'Generate teacher feedback (Model: {model_name})'):
        story = item['story']
        question = item['question']
        student_correct_response = item['student_correct_response']
        student_incorrect_response = item['student_incorrect_response']

        if llm_type == 'openai':
            messages = [{
                    "role": "system",
                    "content": "You are an english education expert."
                    },
                    {"role": "user",
                    "content": construct_teacher_feedback_prompt(story, question, student_incorrect_response, student_correct_response, feedback_gen_mode)
                    }]
            response = get_openai_response(
                                        messages, 
                                        model           = model_name,
                                        temperature     = 1.0,
                                        response_format = {'type':"json_object"},
                                        num_responses   = num_responses
                                        )
            if num_responses == 1:
                try:
                    response = json.loads(response) 
                    if 'feedback' in response:
                        item['teacher_feedback'] = response['feedback']
                    else:
                        item['teacher_feedback'] = response
                except Exception as e:
                    item['teacher_feedback'] = response
            else:
                feedback_list = []
                for r in response:
                    try:
                        r = json.loads(r)
                        if 'feedback' in r:
                            feedback_list.append(r['feedback'])
                        else:
                            feedback_list.append(r)  
                    except Exception as e:
                        feedback_list.append(r)  
                item['teacher_feedback'] = feedback_list

        elif llm_type == 'claude':
            pattern = r'"feedback":\s*"(.*?)"'
            
            system = "You are an english education expert."
            messages = [{
                    "role": "user", 
                    "content": construct_teacher_feedback_prompt(story, question, student_incorrect_response, student_correct_response, feedback_gen_mode)
                    }]
            
            response = get_claude_response(
                                        messages, 
                                        model         = model_name,
                                        temperature   = 1.0,
                                        system        = system,
                                        num_responses = num_responses
                                        )
            if num_responses == 1:
                try:
                    response = response[0].text # Output Format: TextBlock(text='', type='text')
                    match = re.search(pattern, response)
                    if match:
                        feedback = match.group(1)
                    else:
                        feedback = response                        
                    item['teacher_feedback'] = feedback       
                except Exception as e:
                    item['teacher_feedback'] = response  
            else:
                feedback_list = []
                for r in response:
                    try:
                        match = re.search(pattern, r[0].text)
                        if match:
                            feedback = match.group(1)
                        else:
                            feedback = r.text
                        feedback_list.append(feedback)
                    except Exception as e:
                        feedback_list.append(r)
                item['teacher_feedback'] = feedback_list

        elif llm_type == 'open-source':
            if model_name.startswith("google/gemma"):
                messages = [
                        {"role": "user",
                        "content": construct_teacher_feedback_prompt(story, question, student_incorrect_response, student_correct_response, feedback_gen_mode)
                        }]
            else:
                messages = [{
                        "role": "system",
                        "content": "You are an english education expert."
                        },
                        {"role": "user",
                        "content": construct_teacher_feedback_prompt(story, question, student_incorrect_response, student_correct_response, feedback_gen_mode)
                        }]

            prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False, 
                    add_generation_prompt=True
                )

            response = get_huggingface_response(
                                        prompt, 
                                        terminators   = terminators,
                                        pipeline      = pipeline,
                                        model         = model_name,
                                        temperature   = 1.0,
                                        num_responses = num_responses
                                        )
            pattern = r'"feedback":\s*"(.*?)"'
            if num_responses == 1:
                try:
                    response = json.loads(response) 
                    if 'feedback' in response:
                        item['teacher_feedback'] = response['feedback']
                    else:
                        item['teacher_feedback'] = response
                except Exception as e:                    
                    try:
                        match = re.search(pattern, response)
                        if match:
                            feedback = match.group(1)
                        else:
                            feedback = response   
                        item['teacher_feedback'] = feedback    
                    except Exception as inner_e:
                        item['teacher_feedback'] = response         
            else:
                feedback_list = []
                for r in response:
                    try:
                        r = json.loads(r)
                        if 'feedback' in r:
                            feedback_list.append(r['feedback'])
                        else:
                            feedback_list.append(r)  
                    except Exception as e:
                        try:
                            match = re.search(pattern, r)
                            if match:
                                feedback = match.group(1)
                            else:
                                feedback = r
                            feedback_list.append(feedback)
                        except Exception as inner_e:
                            print(f"Error processing response: {inner_e}")
                            feedback_list.append(r)
                item['teacher_feedback'] = feedback_list

    dump_json(result_path, data)
    print(f"[SAVE] File has been successfully saved to {result_path}")


def parser_config():
    """
    Argument Setting
    """
    parser = ArgumentParser()
    parser.add_argument('--feedback_gen_mode', default='w_criteria', type=str, help='w_criteria, wo_criteria')

    # Folder paths
    parser.add_argument('--dataset_name', default='mc160', type=str, help='Name of the dataset to be used (e.g., mc160, mc500).')
    parser.add_argument('--result_folder', default='./result', type=str, help='Path to the folder where the output result files are stored after processing')

    # Model-related parameters
    parser.add_argument('--model_name', default='gpt-4o', type=str, help='Name of the large langauge model to be used (e.g., gpt-4o, claude-3-5-sonnet-20240620, meta-llama/Meta-Llama-3.1-70B-Instruct)') 
    parser.add_argument('--num_responses', default=5, type=int, help='Number of responses')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parser_config()

    # Define paths
    data_folder = f"./data/{args.dataset_name}"
    feedback_folder = f"{args.result_folder}/{args.dataset_name}/feedback/{args.model_name}"
    prepare_dir(feedback_folder)

    input_path = os.path.join(data_folder, 'train.jsonl')
    output_path = os.path.join(feedback_folder, f'train.{args.feedback_gen_mode}.{args.num_responses}.json')
    
    try:
        llm_type_key = get_llm_type(args.model_name)
        print(f"The model '{args.model_name}' belongs to the '{llm_type_key}' LLM type.")
    except ValueError as e:
        print(e)

    generate_student_answer(input_path, input_path)
    generate_teacher_feedback(input_path,
                              output_path,
                              llm_type_key, 
                              args.model_name, 
                              args.feedback_gen_mode, 
                              args.num_responses
                              )
                    
