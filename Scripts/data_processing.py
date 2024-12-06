import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    #Loads the dataset.
    data = pd.read_csv("../Data/creditcard.csv")
    return data

def explore_data(data):
    #Prints dataset info and visualizes class distribution.
    print(data.info())
    print(data.describe())

def class_distribution(data):
    # Visualize class distribution
    sns.countplot(x='Class', data=data)
    plt.title("Class Distribution")
    plt.show()
