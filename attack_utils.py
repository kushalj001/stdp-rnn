import textattack
from textattack.models.wrappers import ModelWrapper
import torch

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, review, device):
    
        self.model = model
        self.review = review # placeholder for dataset: vectors, stoi
        self.device = device

    def __call__(self, text_input_list):
        all_tokens = []
        all_ids = []
        for text in text_input_list:
            tokens = self.review.tokenize(text)
            ids = [self.review.vocab.stoi[tok] for tok in tokens]
            all_tokens.append(tokens)
            all_ids.append(ids)
        text_lengths = [len(tokens) for tokens in all_tokens]
        text_lengths.sort(reverse=True)
        max_seq_len = text_lengths[0]

        padded_text = torch.LongTensor(len(all_tokens), max_seq_len).fill_(1)
        all_ids.sort(key=lambda item: -len(item))
        for i, ids in enumerate(all_ids):
            padded_text[i, :len(ids)] = torch.LongTensor(ids)
        padded_text = padded_text.to(self.device)
        preds = self.model(padded_text, torch.LongTensor(text_lengths))
        preds = torch.sigmoid(preds)
        final_preds = torch.stack([1-preds, preds], dim=1)
        return final_preds.squeeze(-1).detach().cpu().numpy()
            



