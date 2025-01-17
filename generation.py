
# Libraries
import torch
import torch.nn.functional as F
from torch import nn
import math

########################## HELPER FUNCTIONS ######################

def __generate(model, inputs, num_tokens, tokenizer, max_length=50):
    collect = []
    for _ in range(num_tokens):
        output = model(**inputs)
        output_id = torch.argmax(output['logits'][0, -1]).item()
        collect.append(output_id)
        if output_id == tokenizer.eos_token_id or len(collect) >= max_length:
            break
        # Update input_ids and attention_mask
        new_token = torch.tensor([output_id], device=inputs['input_ids'].device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'][0], new_token]).unsqueeze(0)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, 1), value=1)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(collect))


def check_solution(prompt, num_tokens, model_A, model_B, tokenizer, max_length=50):
    print(f"{'>'*20}\n\tPrompt\n{'<'*20}\n{prompt}\n\n")
    
    model_inputs = tokenizer(prompt, return_tensors='pt')
    
    try:
        print(f"{'>'*30}\n\tModel_A Generation\n{'<'*30}")
        print(__generate(model_A, model_inputs, num_tokens, tokenizer, max_length))
    except Exception as e:
        print(f"Error with Model_A: {e}")
    
    try:
        model_inputs = tokenizer(prompt, return_tensors='pt')
        print(f"\n\n{'>'*30}\n\tModel_B Generation\n{'<'*30}")
        print(__generate(model_B, model_inputs, num_tokens, tokenizer, max_length))
    except Exception as e:
        print(f"Error with Model_B: {e}")

