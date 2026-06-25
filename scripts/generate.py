import torch
from models.model import GPT
from tokenization.bpe import BpeTokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # Токенизация промпта
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Инициализация кэша
    past_key_values = None
    
    generated = idx
    
    for _ in range(max_new_tokens):
        # Если кэш есть, подаем только последний токен
        input_idx = generated[:, -1:] if past_key_values is not None else generated
        
        with torch.no_grad():
            logits, past_key_values = model(input_idx, past_key_values=past_key_values, use_cache=True)
            
        # Сэмплирование
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Остановка, если встретили EOS (зависит от вашего токенизатора)
        if next_token.item() == tokenizer.tokenizer.token_to_id("</s>"):
            break
            
    return tokenizer.decode(generated[0].tolist())

# В Colab:
# model = GPT.load_from_checkpoint("checkpoints/best_gqa_model.ckpt", config=config)
# print(generate_text(model, bpe_tok, "The research of neural architectures is"))