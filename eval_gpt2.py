import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Next steps:
# make sure gpt2 encodes <eos>, or have its surprisal rating be computed. refer to previous implementation to see 
# if the surprisal was just given 0.0 for surprisal.surprisal
# fix gpt4 to give solution with full tokens rather than smaller tokens
# fix printing to gpt2_output.tsv
# add items.tsv to combined_result2.csv
#     - foster 1-1 relationship but fixing tokenizing

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--seed', type=int, 
                    default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--temperature', type=float, 
                    default=1.0, help='temperature - higher will increase diversity')
parser.add_argument('--outf', type=str, 
                    default='generated.txt', help='output file for generated text')
parser.add_argument('--prefixfile', type=str, 
                    default='', help='File with sentence prefix from which to generate continuations')
parser.add_argument('--surprisalmode', type=bool, 
                    default=False, help='Run in surprisal mode; specify sentence with --prefixfile')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

if args.cuda:
    model.cuda()

# Load prefix from file and tokenize
with open(args.prefixfile, 'r') as f:
    prefix = f.read().strip()
prefix_tokens = tokenizer.encode(prefix, return_tensors="pt")
if args.cuda:
    prefix_tokens = prefix_tokens.cuda()


def compute_surprisal(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    surprisals = [-torch.log(probabilities[i, input_ids[0, i]]).item() for i in range(input_ids.size(1))]
    return surprisals


if args.surprisalmode:
    with open(args.prefixfile, 'r') as f:
        sentences = f.read().strip().split("<eos>")

    device = torch.device("cuda" if args.cuda else "cpu")
    with open(args.outf, 'w') as outf:
        for sentence in sentences:
            # Tokenize the sentence
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            input = torch.tensor(tokens[:-1], dtype=torch.long).to(device).unsqueeze(0)
            
            # Initialize past to None (GPT-2 uses past for faster sequential generation)
            past = None
            totalsurprisal = 0.0
            
            # Write the first token's surprisal as 0 (as it's the starting point)
            outf.write(tokenizer.decode(tokens[0]) + "\t0.00\n")
            
            # Iterate over tokens in the sentence
            for idx in range(len(tokens) - 1):
                with torch.no_grad():
                    # Get the model's predictions
                    outputs = model(input[:, idx].unsqueeze(0), past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                
                # Calculate the surprisal
                probs = torch.nn.functional.softmax(logits, dim=-1)
                surprisal = -torch.log2(probs[0, tokens[idx+1]]).item()
                
                # Write the token and its surprisal to the output file
                outf.write(tokenizer.decode(tokens[idx+1]) + "\t" + str(surprisal) + "\n")


# sentences = ["The cat sat on the", "It was a sunny day"]
# for sentence in sentences:
#     surprisals = compute_surprisal(sentence)
#     for word, value in zip(sentence.split(), surprisals):
#         print(f"Surprisal for '{word}': {value:.4f}")
#     print("---")
