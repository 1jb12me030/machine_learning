
import pandas as pd

# Sample data
data = [
    ["Name", "Age", "City"],
    ["John", 25, "New York"],
    ["Alice", 30, "Los Angeles"],
    ["Bob", 22, "Chicago"],
    ["Eva", 28, "San Francisco"],
    ["Daniel", 35, "Seattle"],
    ["Grace", 27, "Boston"],
    ["Michael", 32, "Houston"],
    ["Olivia", 26, "Austin"],
    ["William", 29, "Denver"],
    ["Sophia", 24, "Miami"],
    ["Liam", 31, "Atlanta"],
    ["Emily", 23, "Dallas"],
    ["Logan", 34, "Phoenix"],
    ["Ava", 33, "Philadelphia"],
    ["Mason", 28, "San Diego"],
    ["Harper", 29, "Minneapolis"],
    ["Ethan", 26, "Portland"],
    ["Emma", 30, "Detroit"],
    ["Oliver", 27, "Las Vegas"]
]
df=pd.DataFrame(data) #To create dataframe.

df.to_csv("data.csv") #To create csv file

# Specify the file name
# file_name = "sample_data.csv"

# # Write data to CSV file
# with open(file_name, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data)

# print(f"CSV file '{file_name}' created successfully.")
