import torch

def generate_batch(model, tokenizer, batch, device):
    """
    Generate one sentence per prompt in the batch.
    Returns a list of decoded strings (one per prompt).
    """

    # Table 11
    BEAM_SIZE = 10              # Keeps 10 candidate sequences at every step     
    LENGTH_PENALTY = 0.9        # Controls preference for length: slightly favors longer outputs
    NO_REPEAT_NGRAM = 4         # Prevents repeating any 4-word sequence
    MAX_NEW_TOKENS = 128        # Hard limit on generation length
    GEN_BATCH_SIZE = 8
    
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)

    # Use computed prompt lengths to later remove the original prompt portion from the outputs
    prompt_lengths = attention_mask.sum(dim=1)

    # Outputs: [prompt tokens + generated tokens]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=BEAM_SIZE,              
            length_penalty=LENGTH_PENALTY,   
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        # Remove prompt part
        new_tokens = output[prompt_lengths[i]:]

        # Decode
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text.strip())
    return results