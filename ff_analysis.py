import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from datetime import datetime
from fastai.tabular.all import *

sns.set_theme()

df = pd.read_csv('ff_data_2021.csv')
df_origianl = df.copy()
print(df.head())

# Drop the Initial Summary Column for now since it full of supplimentary and not needed information regarding the incident.
df = df.drop(columns=['Initial summary'])

columns = list(df.columns)
fixed_cols = []

# Format the column names to be lowercase and have a underscore instead of spaces
for col in columns:
    col = col.lower()
    col = col.replace(' ', '_')
    fixed_cols.append(col)

df.columns = fixed_cols

# Convert the dates to datetime for easy manipulation
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['date_of_death'] = pd.to_datetime(df['date_of_death'])

# Fill na values with unknown
df['age'].fillna(-99, inplace=True)
df['rank'].fillna('unknown', inplace=True)
df['classification'].fillna('unknown', inplace=True)
df['cause_of_death'].fillna('unknown', inplace=True)
df['nature_of_death'].fillna('unknown', inplace=True)
df['activity'].fillna('Unknown', inplace=True)
df['duty'].fillna('Unknown', inplace=True)
df['property_type'].fillna('unknown', inplace=True)

# Fill the one na value with an estimate based off of the notes in the data
df['incident_date'].fillna(pd.Timestamp('2003-02-03'), inplace=True)

# Returns a list of two strings
def split_col(rank):
    return rank.split('/')[:]

# Returns a single string switched in order
def switch_join_col(rank):
    return '/'.join([rank[1], rank[0]])

# Returns a string in a uniform format
def uniform_col(rank):
    rank = rank.lower()
    rank = rank.replace(' / ', '/')
    rank = rank.replace(' /', '/')
    rank = rank.replace('/ ', '/')
    rank = rank.replace(' - ', '-')
    rank = rank.replace(' -', '-')
    rank = rank.replace('- ', '-')
    rank = rank.replace('-', '/')
    rank = rank.replace('2', 'ii')
    rank = rank.replace('3', 'iii')
    rank = rank.replace('firefigher', 'firefighter')
    rank = rank.replace('(contract)', '')
    rank = rank.replace('  ', '')
    rank = rank.replace('acting ', '')
    rank = rank.replace('fire chief', 'chief')
    rank = rank.replace('chief', 'fire chief')
    rank = rank.replace('ex/captain', 'captain')
    return rank

# Returns a string in alphabetical order
def order_col(rank):
    if '/' in rank:
        temp = split_col(rank)
        if temp[0][0] > temp[1][0]:
            rank = switch_join_col(temp)
    return rank

# Returns a string with applied conditional switching
def conditional_col(rank):
    if '/' in rank:
        temp = split_col(rank)
        if temp[0] == 'firefighter' and temp[1] != 'emt' and temp[1] != 'paramedic':
            rank = ''.join([temp[0]])
    return rank

def firefighter_col(rank):
    if '/' in rank:
        temp = split_col(rank)
        if temp[1] == 'firefighter':
            rank = switch_join_col(temp)
    return rank

def lieutenant_col(rank):
    if '/' in rank:
        temp = split_col(rank)
        if temp[1] == 'lieutenant':
            rank = switch_join_col(temp)
    return rank

# Returns the column in a uniform format
def finalize_col(rank):
    rank = rank.title()
    rank = rank.replace('Iii', 'III')
    rank = rank.replace('Ii', 'II')
    rank = rank.replace('Emt', 'EMT')
    rank = rank.replace('Co/Pilot', 'Co-Pilot')
    rank = rank.replace('Captain ', 'Captain')
    return rank

unique_ranks_prior = len(df['rank'].value_counts())
print(f'There are {unique_ranks_prior} unique ranks in the dataset prior to combining groups.')
print(df['rank'].value_counts()[:10])

# Apply these formating functions using list comprehensions
df['rank'] = [uniform_col(x) for x in df['rank']]
df['rank'] = [order_col(x) for x in df['rank']]
df['rank'] = [firefighter_col(x) for x in df['rank']]
df['rank'] = [lieutenant_col(x) for x in df['rank']]
df['rank'] = [conditional_col(x) for x in df['rank']]
df['rank'] = [finalize_col(x) for x in df['rank']]

unique_ranks_post = len(df['rank'].value_counts())
print(f'There are {unique_ranks_post} unique ranks in the dataset after combining groups.')
print(df['rank'].value_counts()[:10])

# 9/11 only dataset
df_911 = df.loc[df['incident_date'] == pd.Timestamp('2001-09-11')]
df_911 = df_911.loc[df_911['emergency'].str.lower() == 'yes']

# Drop the 9/11 fatalities from the dataset and save a copy off all fatalities for later use
df_all = df.copy()
df = df.drop(df[(df['incident_date'] == pd.Timestamp('2001-09-11')) * (df['emergency'].str.lower() == 'yes')].index)

percent_911 = round(len(df_911) / len(df_all), 2) * 100
percent_rest = 100 - percent_911

print(f'{percent_911}% of firefighter fatalities occured on 9/11.')

labels = ['', '9/11/2001']
sizes = [percent_rest, percent_911]

explode = (0, 0.1)
colors = ['#66b3ff','#ff9999']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
ax1.set_title('Fatalities of 9/11', fontweight='bold', fontsize=18)

plt.tight_layout()
plt.show()