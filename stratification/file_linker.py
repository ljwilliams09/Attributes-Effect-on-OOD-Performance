import pandas as pd
count = 0

input_file = "./prompts/wine_descriptive_prompts.csv"
df = pd.read_csv(input_file)

def generate_filename(row):
    global count
    filename = f"wine_descriptive_{count}.png"
    count += 1
    return filename

df['filename'] = df.apply(generate_filename, axis=1)

# Write the updated DataFrame back to the original CSV
df.to_csv(input_file, index=False)
