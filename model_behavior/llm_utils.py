import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, set_seed, pipeline
from peft import PeftModel
import torch
import pdb
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

#from openai import OpenAI
# import tiktoken
#import google.generativeai as genai
#import anthropic

api_key = os.getenv("OPENAI_API_KEY")

def free_gpu_memory(items_to_delete):
    for item in items_to_delete:
        del item
    torch.cuda.empty_cache()


def move_to_device(model):
    # Move model to the correct device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Moving model to device: {device}")
    if model.device.type != device:
        model.to(device)


def get_model_size_B(model_name: str, default: int = 2) -> int:
    """Get the model size from the model name, in Billions of parameters.
    """
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>\d+)[bB]", model_name)
    if regex:
        return int(regex.group("size")) * int(regex.group("times") or 1)
    else:
        print(f"Could not infer model size from name '{model_name}'")
    return default


def load_nli_model_and_tokenizer(model_name_or_path, config):
    downloaded_nli_model_path = Path(
        config.DOWNLOADED_MODELS_PATH, model_name_or_path.replace("/", "--"))
    # check if nli_model_name path exists
    if downloaded_nli_model_path.exists():
        model_path = downloaded_nli_model_path
    else:
        model_path = model_name_or_path.replace("--", "/")

    if model_name_or_path in ["google--t5_xxl_true_nli_mixture"]:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    elif model_name_or_path in ["potsawee--deberta-v3-large-mnli"]:
        from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
    elif model_name_or_path in ["MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"nli model '{model_name_or_path}' not supported")

    model.eval()  # TODO: is this necessary?
    # Move model to cuda if available
    move_to_device(model)

    return model, tokenizer


class ChatBot:
    def __init__(self, dic_args, config=None, api_key=None, lora_weights_path=None, seed=None):
        args = [dic_args, config, api_key, lora_weights_path]
            
        if dic_args["model_name"] in ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06"]:
            self.chatbot = _OpenAI(*args)
        elif dic_args["model_name"] in ["gemini-1.5-flash", "gemini-1.5-flash-002", "gemini-1.5-pro", "gemini-1.5-pro-002"]:
            self.chatbot = _Gemini(*args)
        elif dic_args["model_name"] in ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]:
            self.chatbot = _Anthropic(*args)
        else:
            self.chatbot = _HFModel(*args)
            # raise ValueError(f"chatbot"+{dic_args["model_name"]}+ "not supported")

    def update_seed(self, seed):
        if seed is not None:
            self.seed = seed

    def initialize_prompt_history(self, prompt, model_name_or_path=None):
        # check if prompt is a string or a list of messages

        if isinstance(prompt, str):
            if model_name_or_path == "HuggingFaceTB/SmolLM3-3B":
                return [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": prompt}
                        ] 
            else:
                return [{"role": "user", "content": prompt}]
 
        elif isinstance(prompt, list):
            return [self.initialize_prompt_history(p) for p in prompt]
        elif isinstance(prompt, dict):
            if "role" not in prompt:
                prompt["role"] = "user"
            elif prompt["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {prompt['role']}")
            if set(prompt.keys()) != {"role", "content"}:
                raise ValueError(f"Prompt has invalid keys: {list(prompt.keys())}. Only 'role' and 'content' are allowed.")
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def get_nr_of_tokens(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_max_context_length(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_temperature_range(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_temperature(self, temperature):
        if temperature is None:
            return None
        min_temperature, max_temperature = self.get_temperature_range()
        if temperature == 'min':
            return min_temperature
        elif temperature == 'max':
            return max_temperature
        elif temperature > max_temperature:
            raise ValueError(f"Temperature {temperature} is higher than the maximum allowed temperature {max_temperature} (min temperature: {min_temperature}).")
        else:
            return temperature


class _HFModel(ChatBot):
    def __init__(self, dic_args, config=None, api_key=None, seed=None):
        if seed is not None:
            set_seed(seed)
        elif config is not None and hasattr(config, "SEED") and config.SEED is not None:
            set_seed(config.SEED)

        if api_key is None:
            api_key = os.getenv('HF_ACCESS_TOKEN')

        model_name_or_path = dic_args["model_name"]
        local_model_directory = dic_args["local_model_directory"]  # default path
        print(">>>> MODEL name or path:", model_name_or_path)
        print(">>>> LORA weights path:", dic_args.get("lora_weights_path", None))
        print(">>>> LOCAL model directory:", local_model_directory)

        self.model, self.tokenizer = self.load_model_and_tokenizer(
            model_name_or_path,
            token=api_key,
            local_model_directory=local_model_directory,
            lora_weights_path=dic_args.get("lora_weights_path", None)
        )

    def load_model_and_tokenizer(self, model_name_or_path, token, local_model_directory, lora_weights_path=None):
        # Check if model exists in local cache, if not download it
        print("Try loading models")
        try:
            # Try to load tokenizer first to check if model exists locally
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=local_model_directory,
                trust_remote_code=True
            )
        except OSError:
            print(f"Model not found in local cache. Downloading {model_name_or_path}...")
            # Download both model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=local_model_directory,
                device_map='auto' if lora_weights_path is None else 'cpu',
                output_hidden_states=False,
                return_dict_in_generate=False,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=local_model_directory,
                trust_remote_code=True
            )
        else:
            # Model exists locally, load normally
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=local_model_directory,
                device_map='auto' if lora_weights_path is None else 'cpu',
                output_hidden_states=False,
                return_dict_in_generate=False,
                trust_remote_code=True
            )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map='auto',  # if lora_weights_path is None else 'cpu',
        )
        print(f'NO LORA, original model loaded to: {pipe.model.device}')
      
        # In case a path for LoRA weights is passed, load the weights and put the
        # original model and the weights together.
        if lora_weights_path is not None:
            from accelerate.utils import get_balanced_memory, infer_auto_device_map
            from accelerate import dispatch_model

            print(f'With LORA, original model loaded to: {pipe.model.device}')
            print(f'Loading LoRA weights from: {lora_weights_path}')

            lora_fine_tuned_model = PeftModel.from_pretrained(
                pipe.model,
                lora_weights_path,
                dtype=getattr(pipe.model, "dtype", None),
            )

            # Dispatch the LoRA fine-tuned model to GPU with `accelerate` (with
            # sharding).
            # Infer a device map for the *wrapped* model
            max_mem = get_balanced_memory(
                lora_fine_tuned_model,
                dtype=getattr(lora_fine_tuned_model, "dtype", None),
                no_split_module_classes=getattr(lora_fine_tuned_model, "_no_split_modules", None),
            )

            devmap = infer_auto_device_map(
                lora_fine_tuned_model,
                max_memory=max_mem,
                no_split_module_classes=getattr(lora_fine_tuned_model, "_no_split_modules", None),
            )

            # Dispatch to devices (and optionally an offload dir for CPU parts)
            lora_fine_tuned_model = dispatch_model(
                lora_fine_tuned_model,
                device_map=devmap,
                # offload_dir="/data1/offload"
            )

            pipe.model = lora_fine_tuned_model

        # Check which devices the model is using.
        model_devices = np.unique([p.device.type for p in pipe.model.parameters()]).tolist()
        print(f'Final model loaded on devices: {model_devices}')

        self._set_pad_token_id(pipe.tokenizer)

        return pipe.model, pipe.tokenizer
    

    def _set_pad_token_id(self, tokenizer):
        """Add end-of-sentence pad token to the model and tokenizer if it doesn't already exist.
        """
        if tokenizer.pad_token_id is None:
            # If there's no pad_token_id, default to eos_token_id or any value of your choice
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def update_seed(self, seed):
        if seed is not None:
            set_seed(seed)

    def __call__(self, prompt, model_name_or_path, decoding_params, num_return_sequences=1, seed=None):
        self.update_seed(seed)

        if model_name_or_path == "tiiuae/falcon-7b-instruct":
                prompt = [f"User: {prompt}\nAssistant:"]
                tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False
            )
        elif any(key in model_name_or_path.lower() for key in ("sft", "dpo", "instruct")) or model_name_or_path == "HuggingFaceTB/SmolLM3-3B":
            prompt = self.initialize_prompt_history(prompt, model_name_or_path)
            prompt = self.ensure_prompt_is_compatible_with_template(prompt)
            tokens = self.tokenizer.apply_chat_template(
                prompt, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)  
        else:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False
            )

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        prompt_length = tokens['input_ids'].shape[1]

        # # run the following to check the exact prompt being used:
        # print(self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False))

        generation_config = {
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_config.update(decoding_params)

        temperature = generation_config.get("temperature", None)
        if temperature is not None:
            temperature = self.get_temperature(temperature)
            if float(temperature) == 0.0:
                generation_config["do_sample"] = False
                generation_config["temperature"] = None
                generation_config["top_p"] = None
            else:
                generation_config["do_sample"] = True

        decoding_params = generation_config  # ensure that final parameters are saved

        # print(f"Generating text with decoding parameters: {generation_config}")

        # Generate new tokens
        generated_ids = self.model.generate(
            **tokens,
            **generation_config,
        )

        # print(f"\n\n\n\n   Generated text:\n{self.tokenizer.decode(generated_ids[0])}\n\n\n\n")

        if num_return_sequences == 1:
            return self.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
        else:
            return [
                self.tokenizer.decode(
                    generated_id[prompt_length:], skip_special_tokens=True)
                for generated_id in generated_ids
            ]

    def ensure_prompt_is_compatible_with_template(self, prompt):
        """
        Ensure the input prompt is compatible with the chat template.
        Fixes issues with unsupported roles or improper role alternation.
        """
        def is_compatible(test_prompt):
            """
            Check if the given prompt is compatible with the chat template.
            Returns True if compatible, False otherwise.
            """
            try:
                _ = self.tokenizer.apply_chat_template(
                    test_prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True
                )
                return True
            except Exception as e:
                # print(f"Compatibility check failed: {e}")
                return False

        # Step 1: Replace 'system' role with 'user' if needed
        test_prompt1 = [{'role': 'system', 'content': 'Hello!'}]
        if not is_compatible(test_prompt1):
            # print("Replacing system role with user role in the prompt.")
            # print(f"Example prompt before: {test_prompt1}")
            # print(f"Example prompt after: {self.replace_system_with_user(test_prompt1)}")
            prompt = self.replace_system_with_user(prompt)

        # Step 2: Concatenate consecutive roles if needed
        test_prompt2 = [{'role': 'user', 'content': 'Hello!'},
                        {'role': 'user', 'content': 'Hello again!'}]
        if not is_compatible(test_prompt2):
            # print("Concatenating consecutive user/assistant roles in the prompt.")
            # print(f"Example prompt before: {test_prompt2}")
            # print(f"Example prompt after: {self.concat_consecutive_roles(test_prompt2)}")
            prompt = self.concat_consecutive_roles(prompt)

        return prompt

    def replace_system_with_user(self, prompt):
        prompt = [{"role": "user" if p["role"] == "system" else p["role"],
                   "content": p["content"]} for p in prompt]
        return prompt

    def concat_consecutive_roles(self, prompt):
        """
        Concatenate consecutive user or assistant roles in the prompt.
        """
        concatenated_prompt = []
        current_role = None
        accumulated_content = ""

        for message in prompt:
            if message["role"] == current_role:
                # Accumulate content if the role is the same as the previous one
                accumulated_content += "\n" + message["content"]
            else:
                if current_role is not None:
                    # Append the previous accumulated message
                    concatenated_prompt.append(
                        {"role": current_role, "content": accumulated_content.strip()})
                # Start accumulating content for the new role
                current_role = message["role"]
                accumulated_content = message["content"]

        # Append the last accumulated message
        if current_role is not None:
            concatenated_prompt.append(
                {"role": current_role, "content": accumulated_content.strip()})

        return concatenated_prompt

    def get_nr_of_tokens(self, input_text):
        return len(self.tokenizer(input_text, return_tensors="pt").input_ids.squeeze())

    def get_max_context_length(self):
        return get_context_window(self.model)

    def get_temperature_range(self):
        # 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability.
        # default: 1.0
        return (0.0, 100.0)

    def _is_bf16_compatible(self) -> bool:
        """Checks if the current environment is bfloat16 compatible."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


class _OpenAI(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        self.api_key = api_key or self.get_api_key()
        self.client = self.initialize_client()
        self.tokenizer = None

    def get_api_key(self):
        return os.getenv('OPENAI_API_KEY')

    def initialize_client(self):
        return OpenAI(api_key=self.api_key)

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]
        generation_config = {
            "model": self.model_name,
            "messages": prompt,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "n": num_return_sequences,
        }
        if temperature is not None:
            generation_config["temperature"] = self.get_temperature(
                temperature)
        if do_sample:
            if top_p is not None:
                generation_config["top_p"] = top_p

        response = self.client.chat.completions.create(
            **generation_config
        )
        # print(f'temperature: {generation_config["temperature"]}')
        # TODO: return the full response object?
        if num_return_sequences == 1:
            full_response_object = response.to_dict()
            response_text = response.choices[0].message.content
            return full_response_object, response_text
        else:
            # TODO
            pass

    def get_nr_of_tokens(self, input_text, model=None):
        """Return the number of tokens used by a list of messages."""
        if model is None:
            model = self.model_name
        # make sure the input_text is of type list and format the prompts
        input_text = self.initialize_prompt_history(input_text)
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print(
                "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.get_nr_of_tokens(input_text, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.get_nr_of_tokens(input_text, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""get_nr_of_tokens() is not implemented for model {model}."""
            )
        num_tokens = 0
        for message in input_text:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_max_context_length(self):
        if self.model_name in [
            'gpt-4o',
            'gpt-4o-2024-05-13',
            'gpt-4o-2024-08-06',
            'chatgpt-4o-latest',
            'gpt-4o-mini',
            'gpt-4o-mini-2024-07-18',
            'gpt-4-turbo',
            'gpt-4-turbo-2024-04-09',
            'gpt-4-turbo-preview',
            'gpt-4-0125-preview',
            'gpt-4-1106-previe',
        ]:
            return 128000
        elif self.model_name in [
            'gpt-4',
            'gpt-4-0613',
            'gpt-4-0314',
        ]:
            return 8192
        elif self.model_name in [
            'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-1106',
        ]:
            return 16385
        elif self.model_name in [
            'gpt-3.5-turbo-instruct',
        ]:
            return 4096
        else:
            raise NotImplementedError(
                f"""get_max_context_length() is not implemented for model {
                    self.model_name}."""
            )

    def get_temperature_range(self):
        # default: 1.0
        return (0.0, 2.0)


class _FMAPISwissAI(_OpenAI):
    def get_api_key(self):
        return os.getenv('FMAPI_SWISSAI_API_KEY')

    def initialize_client(self):
        return OpenAI(api_key=self.api_key, base_url="https://fmapi.swissai.cscs.ch")


class _Anthropic(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]
        generation_config = {
            "model": self.model_name,
            "messages": prompt,
            "max_tokens": max_new_tokens,
        }
        if num_return_sequences > 1:
            raise ValueError("num_return_sequences>1 is not supported")
        if temperature is not None:
            generation_config["temperature"] = self.get_temperature(
                temperature)
        if do_sample:
            if top_p is not None:
                generation_config["top_p"] = top_p
        system_instruction = [p["content"]
                              for p in prompt if p["role"] == "system"]
        if len(system_instruction) > 0:
            # concatenate all system instructions
            system_prompt = " ".join(system_instruction)
            print(f"  System instruction: {system_prompt}")
            generation_config["system"] = system_prompt
        message = self.client.messages.create(
            **generation_config
        )
        # TODO: return the full response object?
        if num_return_sequences == 1:
            return message.content
        else:
            # TODO
            pass

    def get_nr_of_tokens(self, input_text, model=None):
        """Return the number of tokens used by a list of messages."""
        # TODO: check if this is correct
        pdb.set_trace()
        if model is None:
            model = self.model_name
        # make sure the input_text is of type list and format the prompts
        input_text = self.initialize_prompt_history(input_text)
        response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model_name,  # "claude-3-5-sonnet-20241022",
            system="You are a scientist",
            messages=input_text,
        )
        token_count_response = response.json()
        print(f"Token count response: {token_count_response}")
        return token_count_response["input_tokens"]

    def get_max_context_length(self):
        # TODO: implement this
        raise NotImplementedError("Not yet implemented for Anthropic models.")

    def get_temperature_range(self):
        # default: 1.0
        return (0.0, 1.0)


class _Gemini(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        self.model_info = genai.GenerativeModel(model_name_or_path)
        self.model, self.temperature, self.max_new_tokens, self.system_instruction = self.initialize_model_and_generation_config(
            self.model_name)

    def initialize_model_and_generation_config(self, model_name, temperature=None, top_p=None, top_k=None, max_new_tokens=10, system_instruction=[]):
        generation_config = {
            "top_p": top_p,
            "max_output_tokens": max_new_tokens,
            # "candidate_count": num_return_sequences # num_return_sequences>1 is not supported in the current version
        }
        if temperature is not None:
            temperature = self.get_temperature(temperature)
            generation_config.update({"temperature": temperature})
        if top_k is not None:
            generation_config["top_k"] = top_k
        if top_p is not None:
            generation_config["top_p"] = top_p
        if model_name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro-002"]:
            # if model_name in ["gemini-1.5-flash-001", "gemini-1.5-pro-001", "gemini-1.0-pro-002", "gemini-1.5-flash-002"]:
            # seed is a preview feature that is only available for few models
            # by default, a random seed value is used
            generation_config['seed'] = self.seed

        # TODO: mode generation_config to inferencing
        if len(system_instruction) == 0:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
            )
        else:
            # concatenate all system instructions
            system_instruction = " ".join(system_instruction)
            # print(f"  System instruction: {system_instruction}")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
        return model, temperature, max_new_tokens, system_instruction

    def _role_mapping(self, role):
        if role == "assistant":
            return "model"
        else:
            return role

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        system_instruction = [p["content"]
                              for p in prompt if p["role"] == "system"]
        temperature = self.get_temperature(temperature)
        # if do_sample:
        #     # TODO: is this necessary or is the default top_p value better?
        #     top_p = 1.0
        #     top_k = 1000000
        # else:
        #     top_p = self.get_default_top_p()
        #     top_k = None
        if temperature != self.temperature or max_new_tokens != self.max_new_tokens or system_instruction != self.system_instruction:
            # re-initialize the model with the new generation config
            self.model, self.temperature, self.max_new_tokens, self.system_instruction = self.initialize_model_and_generation_config(
                self.model_name, temperature, top_p, top_k, max_new_tokens, system_instruction)

        user_message = prompt.pop()["content"]
        history = [
            {
                "role": self._role_mapping(p["role"]),
                "parts": p["content"],
            }
            for p in prompt if p["role"] != "system"]

        responses = []
        for _ in range(num_return_sequences):
            if len(history) == 0:
                response = self.model.generate_content(user_message)
            else:
                chat_session = self.model.start_chat(history=history)
                response = chat_session.send_message(user_message)
            responses.append(response.text)

        if num_return_sequences == 1:
            return responses[0]
        else:
            return responses

    def get_nr_of_tokens(self, input_text):
        return self.model.count_tokens(input_text).total_tokens

    def get_max_context_length(self):
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-flash-002"]:
            input_token_limit = 1048576
        elif self.model_name in ["gemini-1.5-pro", "gemini-1.5-pro-002"]:
            input_token_limit = 2097152
        else:
            raise NotImplementedError(
                f"get_max_context_length() is not implemented for model {self.model_name}.")
        return input_token_limit

    def get_temperature_range(self):
        # A temperature of 0 means that the highest probability tokens are always selected. In this case, responses for a given prompt are mostly deterministic, but a small amount of variation is still possible. Higher temperatures can lead to more diverse or creative results.
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
            # default: 1.0
            return (0.0, 2.0)
        elif self.model_name in ["gemini-1.0-pro-vision"]:
            # default: 0.4
            return (0.0, 1.0)
        elif self.model_name in ["gemini-1.0-pro-002"]:
            # default: 1.0
            return (0.0, 2.0)
        elif self.model_name in ["gemini-1.0-pro-001"]:
            # default: 0.9
            return (0.0, 1.0)
        else:
            raise NotImplementedError(
                f"""get_temperature_range() is not implemented for model {
                    self.model_name}."""
            )

    def get_default_top_p(self):
        # The default top_p value depends on the model
        # top_p value range: 0.0 - 1.0
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
            return 0.95
        elif self.model_name in ["gemini-1.0-pro", "gemini-1.0-pro-vision"]:
            return 1.0
        else:
            raise NotImplementedError(
                f"""get_default_top_p() is not implemented for model {
                    self.model_name}."""
            )


def post_process_response(response, options, default_option="n/a"):
    """
    Post-processing function to map a response to a predefined set of options.

    Args:
    response (str): The text response to process.
    options (dict): A dictionary where keys are possible string options and values are corresponding scores.
    default_option (str): The default option to map to if the response does not match any option.

    Returns:
    float: The score corresponding to the processed response.
    """
    # Normalize the response
    response = response.replace("\n", "").lower().strip()

    # Sort options keys by length in descending order
    # check if options is a dictionary
    if isinstance(options, dict):
        sorted_keys = sorted(options.keys(), key=len, reverse=True)

        # Check if the response matches any of the sorted options keys
        for key in sorted_keys:
            if response.startswith(key.lower().strip()):
                return options[key]

        # If no match, return the default option score
        # print(f"warning: {response} not defined")
        # Default score if default_option not in options
        return options.get(default_option, 0.5)

    elif isinstance(options, list):
        options = [o.lower().strip() for o in options]
        if response in options:
            return response
        else:
            return default_option


def get_context_window(model):
    # Different model families use different names for the same field
    typical_fields = ["max_position_embeddings", "n_positions",
                      "seq_len", "seq_length", "n_ctx", "sliding_window"]

    # Check which attribute a given model object has
    context_windows = [getattr(model.config, field)
                       for field in typical_fields if field in dir(model.config)]

    # remove None values
    context_windows = [cw for cw in context_windows if cw is not None]

    # Grab the last one in the list; usually there's only 1 anyway
    if len(context_windows) > 0:
        return context_windows[-1]
    else:
        raise ValueError("Could not find context window size in model config.")


set_seed(42)
