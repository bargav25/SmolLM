from transformers import AutoTokenizer, AutoModelForCausalLM
from config import smolConfig
from model import smolLM
from generation import __generate, check_solution
import torch


# Load tokenizer and reference model
checkpoint = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
reference_model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Initialize smolLM
config = smolConfig()
test_model = smolLM(config)

# Load weights
state_dict = torch.load("BareBones_SmolLM-135M.pt")
test_model.load_state_dict(state_dict, strict=False)

check_solution(prompt="Given the following film movie by a critic, rate it out of 10. Respond in a single number.\n\nThe movie started off extremely well, but just got worse after that.\nThe storyline was all over the place and everyone acted terribly.\n 10/10 would not recommend! \n\n ",
               num_tokens=1,
               model_A=reference_model,
               model_B=test_model, tokenizer=tokenizer)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
print(__generate(test_model, inputs, num_tokens=50, tokenizer=tokenizer))


