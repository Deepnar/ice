"""ICE v2 MicroNER decoding; input tokens are embedded individually."""

import torch


def extract_entities(model, tokenizer, embedder, text: str, max_chars: int | None = None):
    if max_chars is not None:
        text = text[:max_chars]
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    if not encoding["input_ids"]:
        return []
    token_strings = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    embeddings = embedder.encode(token_strings, convert_to_tensor=True, show_progress_bar=False)
    if embeddings.shape[-1] != 384:
        raise ValueError("ICE v2 MicroNER requires exactly 384 embedding coordinates")
    embeddings = embeddings.to(next(model.parameters()).device).float()
    with torch.no_grad():
        predictions = model(embeddings.unsqueeze(0)).argmax(dim=-1).squeeze(0).tolist()
    entities = []
    start = end = None
    for label, (token_start, token_end) in zip(predictions, encoding["offset_mapping"]):
        if label == 0:
            if start is not None:
                entities.append((start, end))
            start, end = token_start, token_end
        elif label == 1 and start is not None:
            end = token_end
        elif start is not None:
            entities.append((start, end))
            start = None
    if start is not None:
        entities.append((start, end))
    glued = []
    for start, end in entities:
        if glued and not text[glued[-1][1]:start].strip():
            glued[-1] = (glued[-1][0], end)
        else:
            glued.append((start, end))
    return [text[start:end].strip() for start, end in glued if text[start:end].strip()]
