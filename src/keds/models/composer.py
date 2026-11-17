"""Differentiable CLIP pseudo-word insertion, with EOT shifted by K-1.

Replace the tokenized placeholder directly instead of re-tokenizing repeated
asterisks, which can produce different BPE sequences. Preserve a 77 token
context and reject overflow instead of silently dropping modification text.
"""
import torch
from torch.nn import functional as F


def inject_tokens(token_embedding, ids, placeholder_id, eos_id, pseudo):
    if pseudo.ndim != 3 or pseudo.shape[0] != ids.shape[0] or pseudo.shape[1] not in (1, 3):
        raise ValueError("Pseudo tokens require [B,1 or 3,D]")
    embedded = token_embedding(ids)
    if pseudo.shape[-1] != embedded.shape[-1]:
        raise ValueError("Pseudo token dimension does not match CLIP text width")
    rows, ends = [], []
    for i in range(len(ids)):
        positions = (ids[i] == placeholder_id).nonzero(as_tuple=True)[0]
        if len(positions) != 1:
            raise ValueError("Each prompt must tokenize to exactly one '*' placeholder")
        pos = int(positions[0])
        eos_positions = (ids[i] == eos_id).nonzero(as_tuple=True)[0]
        if not len(eos_positions):
            raise ValueError("Missing CLIP EOT token")
        end = int(eos_positions[0]) + pseudo.shape[1] - 1
        if end >= ids.shape[1]:
            raise ValueError("Modification exceeds CLIP context after pseudo-word insertion")
        if pos >= int(eos_positions[0]):
            raise ValueError("Placeholder must occur before EOT")
        rows.append(torch.cat([embedded[i, :pos], pseudo[i].to(embedded),
                               embedded[i, pos + 1:]], dim=0)[:ids.shape[1]])
        ends.append(end)
    return torch.stack(rows), torch.tensor(ends, device=ids.device)


def compose(clip, tokenizer, prompts, pseudo, placeholder_id):
    inputs = tokenizer(prompts, padding="max_length", max_length=77,
                       truncation=False, return_tensors="pt")
    ids = inputs["input_ids"].to(pseudo.device)
    if ids.shape[1] != 77:
        raise ValueError("Modification exceeds CLIP 77-token context")
    text = clip.text_model
    embeds, ends = inject_tokens(text.embeddings.token_embedding, ids, placeholder_id,
                                 tokenizer.eos_token_id, pseudo)
    states = embeds + text.embeddings.position_embedding(text.embeddings.position_ids[:, :77])
    # CLIP's EOT cannot attend to padding. Matching official OpenAI CLIP only
    # requires the causal mask; no padding mask is needed for EOT pooling.
    mask = torch.full((77, 77), torch.finfo(states.dtype).min,
                      dtype=states.dtype, device=states.device).triu(1)
    encoded = text.encoder(inputs_embeds=states,
                           causal_attention_mask=mask[None, None].expand(len(ids), 1, 77, 77),
                           return_dict=True)
    hidden = text.final_layer_norm(encoded.last_hidden_state)
    pooled = hidden[torch.arange(len(ids), device=ids.device), ends]
    return F.normalize(clip.text_projection(pooled).float(), dim=-1)

