# add the parent directory to the path
from utils import get_experiment_start_date, format_prompt
from llm_utils import ChatBot
import argparse
from tqdm import tqdm
import csv
from pathlib import Path
import pandas as pd
import os
import re
import time

prompt_formats = ["do you agree or disagree", "between support and oppose", "are you pro or con", "do you support or oppose",
                  "\"favorable\" or \"detrimental", "\"beneficial\" or \"harmful\"", "pro or con", "support or oppose", "\"agree\" or \"disagree\"",
                  "neither \"beneficial\" nor \"harmful\"", "neither \"favorable\" nor \"detrimental\"", "neither \"pro\" nor \"con\""]


def map_response(answer):
    answer = answer.lower().replace("\n", " ")

    for prompt_format in prompt_formats:
        if prompt_format in answer:
            answer = answer.replace(prompt_format, "")

    word_pattern = re.compile(r'[a-z]+')
    words = word_pattern.findall(answer)
    if len(words) == 0:
        return None
    for i, word in enumerate(words):
        if word in {'beneficial', 'advantageous', 'favorable', "agree", "support", "pro"}:
            if i != 0:
                if words[i - 1] == "not":  # if the word is not, then it is negative
                    return 0
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":  # if the word is don't, then it is positive
                            return 0
                    else:
                        return 1
                else:
                    return 1
            else:
                return 1
        if word in {'detrimental', 'disadvantageous', 'harmful', "disagree", "oppose", "con"}:
            if i != 0:
                if words[i - 1] == "not":  # if the word is not, then it is positive
                    return 1
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":  # if the word is don't, then it is positive
                            return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
    return None


def load_all_prompts_from_file(all_prompts_path):
    all_prompts_df = pd.read_csv(all_prompts_path)

    return all_prompts_df


def main(model, api_key, lora_weights_path=None, seed=42, output_folder='results'):
    dic_args = {"model_name": model}
    dic_args["local_model_directory"] = args.local_model_directory
    if lora_weights_path:
        print(f"Using LoRA weights from: {lora_weights_path}")
        dic_args["lora_weights_path"] = lora_weights_path

    # Pass kwargs to avoid positional mixups
    chatbot = ChatBot(dic_args, api_key=api_key, seed=seed).chatbot

    generation_config = {
        'do_sample': False,
        'max_new_tokens': 10,
        'temperature': 1.0,
        'top_p': None,
    }

    os.makedirs(output_folder, exist_ok=True)
    filename = Path(
                output_folder,
                f"{model.split('/')[-1]}_lora_{args.leaning_alignment}_{get_experiment_start_date()}.csv"
            )
    print("Saving on:", filename)

    with open(filename, 'w') as file:
        file.write(
            "prompt_id,sample_id,response_mapped,response,prompt_description_id\n")

    all_prompts = load_all_prompts_from_file(all_prompts_path)

    if args.model == "HuggingFaceTB/SmolLM3-3B":
        prompt_descriptions =  pd.read_csv("data/en_clf_templates_final.csv", index_col=0)
    elif any(keyword in args.model.lower() for keyword in ["instruct", "dpo", "sft"]):
        prompt_descriptions = pd.read_csv("data/en_clf_templates_final.csv", index_col=0)
    else:
        prompt_descriptions =  pd.read_csv("data/en_base_templates_final.csv", index_col=0)

    print("Prompt descriptions:", len(prompt_descriptions))
    print(prompt_descriptions)

    # iterate over all prompt templates/prompt instructions
    for row_index, (_, row_template) in enumerate(prompt_descriptions.iterrows()):

        prompt_template = row_template['template']
        template_id = row_template['template_id']

        # if int(template_id)<=6:
        #   continue

        print(f"Running prompts with template: {prompt_template} (ID: {template_id}), index: {row_index+1}")
        
        for index, row in tqdm(all_prompts.iterrows(), desc="Classifying prompts", total=len(all_prompts)):
            prompt_id = row['ID']
            statement = row['statement']

            prompt = format_prompt(
                prompt_template, statement)

            for sample in list(range(30)): #sampling 30 times with the same prompt
                try:
                    response = chatbot(
                        prompt,
                        args.model,
                        generation_config,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"Error while processing prompt {prompt_id}: {e}")
                    response = f" Could not generate response for prompt {prompt_id} due to error: {e}"
                    exit()

                if args.model == "gpt-4o-2024-08-06":
                    response = response[1]

                response_mapped = map_response(response)
                print(prompt)
                print(response)
                print("\n\n\n")

                # Append the cleaned row along with the empty 'reformulations' column to the new CSV
                with open(filename, mode='a', newline='', encoding='utf-8') as new_file:
                    writer = csv.writer(new_file)
                    writer.writerow(
                        [prompt_id,sample,response_mapped,response,template_id])

    print(
        f"Finished running all queries with model {model} :)")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run statements on offline models using Hugging Face.")
    parser.add_argument('-m', '--model', required=True, type=str,
                        help="Model to run.")
    parser.add_argument('-a', '--api_key', required=False, type=str, default=0,
                        help="Api key (for gemini or openai) or hf token. If not specified, will be loaded from .env file.")
    parser.add_argument('-o', '--output_folder', required=False, type=str, default='results',
                        help="Name of the output folder.")
    parser.add_argument("-d", "--data_path", type=str, default="prob_vaa_en")
    parser.add_argument("-l", '--leaning_alignment', type=str, 
                        default=None,
                        help="Leaning to locate LoRA weights.")
    parser.add_argument("-lp", '--local_model_directory', type=str, default="/data1/shared_models/",
                        help="Random seed for reproducibility.")
    parser.add_argument("--lora_weights_path", type=str, default="/data/milanlp/moscato/bias_training_data")

    args = parser.parse_args()

    tic = time.time()
    all_prompts_path = f"data/{args.data_path}.csv"

    lora_weights_path = None

    if args.leaning_alignment is not None:
        if "SFT" in args.model:
            lora_weights_path = f"{args.lora_weights_path}/aligned_models/lora_adapters/{args.leaning_alignment}_OLMo-2-1124-13B-SFT/"
        elif "DPO" in args.model:
            lora_weights_path = f"{args.lora_weights_path}/aligned_models/lora_adapters/{args.leaning_alignment}_OLMo-2-1124-13B-DPO/"
        elif "Instruct" in args.model:
            lora_weights_path = f"{args.lora_weights_path}/aligned_models/lora_adapters/{args.leaning_alignment}_OLMo-2-1124-13B-Instruct/"
        else:
            lora_weights_path = f"{args.lora_weights_path}/aligned_models/lora_adapters/{args.leaning_alignment}_OLMo-2-1124-13B_noCT/"


    main(args.model, args.api_key, lora_weights_path, output_folder=args.output_folder)
    print("Done!")  
    toc = time.time()
    print(f"Total time taken: {toc - tic:.2f} seconds")


    ##############################
    # RUN EXAMPLES
    ##############################

    # python classification_with_model_prompting.py -m "allenai/OLMo-2-1124-13B-Instruct" -l "left"
    # python classification_with_model_prompting.py -m "gpt-4o-2024-08-06" 
    # python3 classification_with_model_prompting.py -m "allenai/OLMo-2-1124-13B-Instruct" -d "prob_vaa_en" -l "left" -lp "/data1/shared_models/" --lora_weights_path "/data1/shared_models"
