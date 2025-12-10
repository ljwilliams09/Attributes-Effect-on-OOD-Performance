# Regressing Prompt Attributes to VQA Score - help from ChatGPT on this

# ---- Load libraries ----
library(tidyverse)
library(car)
library(broom)
library(MASS)
library(corrplot)

# ---- Read dataset ----
df <- read.csv("./stratification/wine_control_prompts.csv", stringsAsFactors = FALSE)


# ---- Clean + preprocess ----

# Convert categorical fields to factors
df$spatial_constraints <- as.factor(df$spatial_constraints)
df$ambiguity <- as.factor(df$ambiguity)

# Scale numeric features
numeric_vars <- c(
  "word_count",
  "descriptor_words",
  "sentence_count",
  "num_visual_attributes"
)

df[numeric_vars] <- scale(df[numeric_vars])

# # ---- Correlation check ----
# cor_mat <- cor(df[numeric_vars])
# corrplot(
#   cor_mat,
#   method = "color",
#   addCoef.col = "black",
#   tl.cex = 0.8
# )

# ---- Linear Regression ---- Using robust regression
lm_model <- lm(
  vqascore ~ 
    word_count +
    descriptor_words +
    sentence_count +
    num_visual_attributes +
    spatial_constraints +
    ambiguity,
  data = df
)

summary(lm_model)

# ---- Multicollinearity test ----
vif(lm_model)

# ---- Robust regression ---- using robust regression to downweight outliers
robust_model <- rlm(
  vqascore ~ 
    word_count +
    descriptor_words +
    sentence_count +
    num_visual_attributes +
    spatial_constraints +
    ambiguity,
  data = df
)

summary(robust_model)

cat("\n---- OLS SUMMARY ----\n")
print(summary(lm_model))

cat("\n---- VIF ----\n")
print(vif(lm_model))

cat("\n---- ROBUST REGRESSION SUMMARY ----\n")
print(summary(robust_model))