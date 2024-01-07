def extract_disease(content):
    if '병명:' in content:
        return content.split('병명:')[1].split('(')[0].strip()
    else:
        return None

# Apply the function and remove rows where disease name is None
df['disease name'] = df['disease name'].apply(extract_disease)
df.dropna(subset=['disease name'], inplace=True)
df = df[df["disease name"] != "없음"]
