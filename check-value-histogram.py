import matplotlib.pyplot as plt
import pandas as pd

# Plot distribution of values
def plot_value_distribution(df):
    plt.hist(df['value'], bins=50)
    plt.title("Distribution of Test Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

def main():
    # Example: load your normalized dataset
    df = pd.read_csv('normalized-data.csv')
    plot_value_distribution(df)
    
    # Check some negative cases
    negatives = df[df['value'] < 0]
    print(negatives.head(10))
    print(f"Total negative values: {len(negatives)}")

if __name__ == "__main__":
    main()
