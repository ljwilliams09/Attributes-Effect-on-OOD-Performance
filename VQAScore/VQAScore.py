import t2v_metrics

def vqa_score(image_files, key, base_prompt):
    score_func = t2v_metrics.get_score_model(model="gpt-4.1", device="cuda", api_key=key, top_logprobs=20) 
    return score_func(images=image_files, texts=base_prompt).flatten().cpu().tolist()