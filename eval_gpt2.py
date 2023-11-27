import argparse
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, AutoTokenizer, GPT2Config, GPT2Model
import torch.nn.functional as F


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

# special_tokens_dict = {'additional_special_tokens': ['<eos>']}
# num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# configuration = GPT2Config(vocab_size=100000, n_positions=8000)

# model = GPT2Model(configuration)

# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.resize_token_embeddings(len(tokenizer))
model.eval()

# with open(args.prefixfile, 'r') as f:
#     prefix = f.read().strip()
# prefix_tokens = tokenizer.encode(prefix, return_tensors="pt")
# if args.cuda:
#     prefix_tokens = prefix_tokens.cuda()
# print(prefix_tokens)


def compute_surprisal(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    surprisals = [-torch.log(probabilities[i, input_ids[0, i]]).item() for i in range(input_ids.size(1))]
    return surprisals

# if args.surprisalmode:
#     with open(args.prefixfile, 'r') as f:
#         content = f.read().strip()
#         sentences = re.split(r'(?<=<eos>)', content)

#     device = torch.device("cuda" if args.cuda else "cpu")
#     with open(args.outf, 'w') as outf:
#         for sentence in sentences:
#             # Tokenize the sentence
#             tokens = tokenizer.encode(sentence, add_special_tokens=False)
#             input = torch.tensor(tokens[:-1], dtype=torch.long).to(device).unsqueeze(0)
            
#             past = None
#             totalsurprisal = 0.0
            
#             # Write the first token's surprisal as 0 (as it's the starting point)
#             outf.write(tokenizer.decode(tokens[0]) + "\t0.00\n")
            
#             # Iterate over tokens in the sentence
#             for idx in range(len(tokens) - 1):
#                 with torch.no_grad():
#                     # Get the model's predictions
#                     outputs = model(input[:, idx].unsqueeze(0), past_key_values=past)
#                     logits = outputs.logits[:, -1, :]
#                     past = outputs.past_key_values
                
#                 # Calculate the surprisal
#                 probs = torch.nn.functional.softmax(logits, dim=-1)
#                 surprisal = -torch.log2(probs[0, tokens[idx+1]]).item()
                
#                 # Write the token and its surprisal to the output file
#                 outf.write(tokenizer.decode(tokens[idx+1]) + "\t" + str(surprisal) + "\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with open(args.prefixfile, 'r') as f:
    raw_sentences = f.readlines()
sentences = [sentence.replace("<eos>", "<|endoftext|>").strip() for sentence in raw_sentences]

if args.surprisalmode:
    with open(args.outf, 'w') as outf:
        for sentence in sentences:
            torch.manual_seed(args.seed)
            tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
            input_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long).unsqueeze(0).to(device)
            totalsurprisal = 0.0
            first_token_id = tokenized_sentence[0]
            input_ids = torch.cat((input_ids, torch.tensor([[first_token_id]], dtype=torch.long).to(device)), dim=1)
            outf.write(tokenizer.decode([first_token_id]) + "\t0.00\n")
            for token_id in tokenized_sentence[1:]:
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                word_surprisals = -1 * torch.log2(probs)
                word_surprisal = word_surprisals[0, token_id].item()

                token_text = tokenizer.decode([token_id])
                if token_text not in [" ", "er", "ous", "oured"]:
                    outf.write(token_text + "\t" + str(word_surprisal) + "\n")

                # outf.write(tokenizer.decode([token_id]) + "\t" + str(word_surprisal) + "\n")
                input_ids = torch.cat((input_ids, torch.tensor([[token_id]], dtype=torch.long).to(device)), dim=1)
