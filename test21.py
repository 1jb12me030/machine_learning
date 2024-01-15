from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd

def search_reviews(df, product_description, n=3, pprint=True):
    try:
        embedding = get_embedding(product_description, model='text-embedding-ada-002')
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

# Sample data for demonstration purposes
df = pd.DataFrame({
    'ada_embedding': [get_embedding('tasty beans', model='text-embedding-ada-002'),
                      get_embedding('delicious beans', model='text-embedding-ada-002'),
                      get_embedding('awesome beans', model='text-embedding-ada-002')],
    'product_description': ['tasty beans', 'delicious beans', 'awesome beans']
})

df_people = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas']
})

# Searching for reviews and rearranging people data based on search results
res = search_reviews(df, 'delicious beans', n=3)

if res is not None:
    rearranged_people_data = df_people[df_people['Name'].isin(res.index)]
    print(rearranged_people_data)
else:
    print("Unable to get embeddings. Check your API key and network connectivity.")
