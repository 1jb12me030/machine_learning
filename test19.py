import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = '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'

# Your data
data = {
    'Name': [ 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29],
    'City': ['Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas','x','y','z'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Generate embeddings for the 'City' column
embeddings = []
for city in df['City']:
    if city:
        response = openai.Completion.create(
            engine="text-embedding-ada-002",
            prompt=city,
            max_tokens=50,  # Set length to a value greater than 0
            n=1,
            logprobs=0,   # Specify logprobs parameter
            stop=None,
            temperature=0,
        )
        embedding = response['choices'][0]['text']
        embeddings.append(embedding)
    else:
        # Handle empty strings or missing values as needed
        embeddings.append(None)

# Ensure 'City' and 'City_Embeddings' have the same length
df['City_Embeddings'] = embeddings[:len(df['City'])]

# Display the DataFrame with embeddings
print(df[['Name', 'City', 'City_Embeddings']])
