import csv
import gpt_prompter

def main():
    base_prompt = "a glass of red wine"
    system_prompt = "You are a research assistant that is creating prompts for us to test on image generation. You will be given a list of prompt attributes, and a target object for the image generator to create. Return one image prompt that follows the attribute guidelines to elicit the photo from the image generator. Return only the final image prompt with no explanation or extra text. Your main focus should be getting a glass of red wine." \
    "Here are descriptions of what each variable means: " \
    "descriptor_words_count: adjectives + adverbs count" \
    "spatial_constraints: how the object is viewed and in what perspective is it portrayed" \
    "num_visual_attributes: (color words, size words)" \
    "sentence_count: number of sentences" \
    "style: photorealistic" \
    "ambiguity: how ambiguous is the prompt? clear would mean that it is directly indicated that it is a wine glass, ambiguous would mean that the object is indirectly referenced."

    id = 0
    descriptor_words = 3
    sentence_count = 2
    spatial_constraints = ["none", "top", "bottom", "side"] #["none", "top", "bottom", "side", "close_up", "far"]
    visual_attributes = 3
    ambiguity_score = ["clear", "partially ambiguous", "fully ambiguous"]



    combinations = []
    headers = ["prompt_id", "word_count", "descriptor_words", "sentence_count", "num_visual_attributes", "spatial_constraints", "ambiguity", "prompt"]
    for des_words in range(1,descriptor_words + 1):
        for sen_con in range(1, sentence_count + 1):
            for visual in range(1, visual_attributes + 1):
                for constraint in spatial_constraints:
                        for amb_score in ambiguity_score:
                            id += 1
                            combinations.append([id, None, des_words, sen_con, visual, constraint, amb_score])
    
    with open("./wine_control.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in combinations:
            prompt = gpt_prompter.prompt_generation(base_prompt,row,system_prompt)
            assert prompt is not None
            row[1] = len(prompt.split())
            writer.writerow(row + [prompt])

if __name__ == "__main__":
    main()