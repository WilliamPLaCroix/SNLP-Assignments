# First we import torch and get ready to load our model
import torch
import torch.nn.functional as F

# Define a function to get the top word picks from the model's output
def get_top_picks(output, k=10):
    probabilities = F.softmax(output.float(), dim=-1)
    values, indices = torch.topk(probabilities, k)
    top_picks = [(idx.item(), val.item()) for idx, val in zip(indices[0], values[0])]
    return top_picks

def model_predict(input_str: str, model, tokenizer, top_picks: int = 10):
    indexed_tokens = tokenizer(input_str, return_tensors="pt")

    with torch.no_grad():
      prediction = model(**indexed_tokens).logits[:, -1, :]

    # Softmax the logits into probabilities
    probabilities = F.softmax(prediction, dim=-1)

    # Generate next token
    generated_next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
    generated = torch.cat([indexed_tokens.input_ids, generated_next_token], dim=-1)

    # Get result
    result_string = tokenizer.decode(generated.tolist()[0])

    # Get top picks for the next token by the language
    top_picks = get_top_picks(prediction, k=top_picks)
    top_tokens = torch.tensor([top_pick[0] for top_pick in top_picks])
    top_tokens = tokenizer.decode(top_tokens, skip_special_tokens=True).split()
    top_picks = {top_token: prob for top_token, (_, prob) in zip(top_tokens, top_picks)}

    # Print string
    print(top_picks)
    print(result_string)
