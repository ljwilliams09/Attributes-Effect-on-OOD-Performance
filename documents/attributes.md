# Prompt Attributes

- **Core Quantitative Attributes**
  - word_count: total words
  - char_count: characters including or excluding spaces
  - token_count: depends on tokenizer
- **Semantic / Content-Based Attributes**
  - descriptor_words_count: adjectives + adverbs count
  - noun_count
  - verb_count
  - proper_noun_count
  - emotion_words (anger, joy, fear, etc.)
  - sensory_words_count (color terms, shape terms, texture terms)
  - object_count (unique nouns)
  - domain_labels (e.g., food, animals, vehicles)
- **Syntactic Attributes**
  - sentence_count
  - avg_sentence_length
  - complexity_score
  - question_present (boolean)
  - command_present ("show", "generate", "make", etc.)
- **Stylistic Attributes**:
  - ambiguity_score
  - hedging_words_count ("maybe", "sort of", "somewhat")
  - intensifier words_count ("very", "extremely")
  - vividness_score (can be computed using concreteness lexicons)
- **Image-Relevant Attributes**
  - num_visual_attributes (color words, size words)
  - num_count_words ("three dogs", "two glasses")
  - num_actions ("jumping", "riding", "pouring")
  - specificity_score (how detailed the described object is)
- **LLM-Specific Attributes**
  - prompt_length_category (short / medium / long)
  - entropy_of_vocabulary (measures lexical diversity)
  - novel_word_ratio (rare words vs common words)
  - ambiguity_category (clear / partially ambiguous / fully ambiguous)

# Prompt Writing Guide

(Reference: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/img-gen-prompt-guide)

- **Subject**: The first thing to think about with any prompt is the subject: the object, person, animal, or scenery you want an image of.
- **Context and background**: Just as important is the background or context in which the subject will be placed. Try placing your subject in a variety of backgrounds. For example, a studio with a white background, outdoors, or indoor environments.

- **Style**: Finally, add the style of image you want. Styles can be general (painting, photograph, sketches) or very specific (pastel painting, charcoal drawing, isometric 3D).
