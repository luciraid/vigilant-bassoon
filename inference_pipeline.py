import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class InferencePipeline:
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def generate_response(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, prompt):
        return self.generate_response(prompt)
