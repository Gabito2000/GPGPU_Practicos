import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("results.csv")

# Check the column names
print(df.columns)

# Make sure column names are trimmed and have consistent casing
df.columns = df.columns.str.strip().str.lower()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['y'], marker='o', linestyle='-')
plt.title('Plot of y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
