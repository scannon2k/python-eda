# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from IPython.display import display

%matplotlib inline

# to suppress warnings
import warnings
warnings.filterwarnings('ignore')

# load the 'matches' dataset from csv
matches = pd.read_csv('matches.csv')

# observe the rows and columns
rows, cols = matches.shape
print(f"There are {rows} rows and {cols} columns in the matches dataset")

# view the dataset with head()
display(matches.head())

display(matches.describe().T)
display(matches.info())

print("Number of duplicates in matches:", matches.duplicated().sum())

# id zZK6BQrf appears in the dataset 3 times
print("There are duplicate ids in the dataset. For instance, zZK6BQrf appears 3 times:")
display(matches[matches['id']=='zZK6BQrf'])

# print number of matches before dropping duplicates
rows, cols = matches.shape
print(f"There are {rows} rows and {cols} columns in matches before dropping duplicates")

# drop the 429 duplicates in the dataset
matches = matches.drop_duplicates()

# print number of matches before dropping duplicates
rows, cols = matches.shape
print(f"There are {rows} rows and {cols} columns in matches after dropping duplicates")

# check for missing values in the dataset
print(matches.isnull().sum())

# print the total missing values
print("\nTotal missing values:", matches.isnull().sum().sum())

# title and axis labels
plt.title("Rating Distribution")
plt.xlabel('Rating')
plt.ylabel('Frequency')

# plot both rating frequency distributions
plt.hist(matches['white_rating'], bins=35, alpha=0.7)
plt.hist(matches['black_rating'], bins=35, alpha=0.6)
plt.legend(['white', 'black'])

# remove right and top lines on chart
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
ax.grid(False)
ax.set_facecolor("white")

plt.show()

sns.set_style("whitegrid")

# title and axis labels
plt.title("Winner by Color")

# bar chart for white win, black win, or draw
matches.winner.value_counts().plot(kind='bar', rot = 0, alpha=0.8, color='#00429d')
ax = plt.subplot(111)
ax.spines[['top', 'right','left','bottom']].set_visible(True)
#ax.grid(False)
ax.grid(axis='x')
ax.set_facecolor("white")

# display the plot
plt.show()

print("Winner by Color:\n")
print(matches['winner'].value_counts())

# title and axis labels
plt.title("Victory Status")

# bar chart for game outcome: resign, mate, outoftime, draw
matches.victory_status.value_counts().plot(kind='bar', rot = 0, alpha=0.8, color='#00429d')
ax = plt.subplot(111)
ax.spines[['top', 'right','left','bottom']].set_visible(True)
#ax.grid(False)
ax.grid(axis='x')
ax.set_facecolor("white")

# display the plot
plt.show()

display(matches.corr())

# plot seaborn heatmap
plt.figure(figsize=(8,8))
sns.heatmap(matches.corr(), annot=True, cmap='Blues').set_title('Correlation Matrix') #cmap='viridis'

# pairplot will not accept boolean values
# created_at and last_move_at don't give useful results, practically perfectly correlated
df = matches.drop(columns=['rated','created_at','last_move_at'])

sns.pairplot(df, corner=True)

