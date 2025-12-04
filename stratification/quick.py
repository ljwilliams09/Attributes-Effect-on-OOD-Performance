import pandas as pd
count = 0
df = pd.read_csv("./wine_prompts.csv")

def generate_filename(row):
    global count
    filename = f"wine_{count}.png"
    count += 1
    return filename

df['filename'] = df.apply(generate_filename, axis=1)

# Write the updated DataFrame back to the original CSV
df.to_csv("./wine_prompts.csv", index=False)
