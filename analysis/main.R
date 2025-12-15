# Regressing Prompt Attributes to VQA Score - help from ChatGPT on this

# ---- Load libraries ----
library(tidyverse)
library(car)
library(broom)
library(MASS)
library(corrplot)

# ---- Read dataset ----
df <- read.csv("wine_prompts.csv", stringsAsFactors = FALSE)


# ---- Clean + preprocess ----

# Convert categorical fields to factors
df$spatial_constraints <- as.factor(df$spatial_constraints)
df$ambiguity <- as.factor(df$ambiguity)

# Scale numeric features
numeric_vars <- c(
  "word_count",
  "descriptor_words",
  "sentence_count"
  # "num_visual_attributes"
)

df[numeric_vars] <- scale(df[numeric_vars])

# ---- Linear Regression ---- Using robust regression
lm_model <- lm(
  vqascore ~ 
    word_count +
    descriptor_words +
    num_visual_attributes, 
    # spatial_constraints +
    # ambiguity,
  data = df
)
summary(lm_model)