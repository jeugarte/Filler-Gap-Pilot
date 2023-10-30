from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()

def compute_surprisal(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    surprisals = [-torch.log(probabilities[i, input_ids[0, i]]).item() for i in range(input_ids.size(1))]
    return surprisals

sentences = ["The cat sat on the", "It was a sunny day"]
for sentence in sentences:
    surprisals = compute_surprisal(sentence)
    for word, value in zip(sentence.split(), surprisals):
        print(f"Surprisal for '{word}': {value:.4f}")
    print("---")
