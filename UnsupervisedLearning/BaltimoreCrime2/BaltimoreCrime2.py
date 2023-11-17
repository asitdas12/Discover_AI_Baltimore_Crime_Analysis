# Add all needed libraries to work on your data set
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
# %matplotlib inline
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Read .csv file
df = pd.read_csv('UnsupervisedLearning/BaltimoreCrime2/BaltimoreCrimeData2.csv')

# Write code to inspect the data frame
df

# Write code to get information about null values in the data frame
df.info()

# Check for missing values
df.isnull().sum()

# Drop unwanted columns in the data set
dfEdit = df
dfEdit.drop(['CrimeDateTime', 'District', 'Inside_Outside', 'Post', 'GeoLocation', 'X', 'Y', 'RowID', 'Location', 'CrimeCode', 'Weapon', 'Premise', 'VRIName', 'Total_Incidents', 'Shape', 'Neighborhood'], 
axis=1, inplace=True)

# Check the summary again to see if there are no unwanted columns remaining
dfEdit.info()

# Write code to inspect statistical information about the data set
dfEdit.describe()

# View the values of the 'Description' column
dfEdit['Description'].unique()

# Delete all rows that correspond to crimes other than 'HOMICIDE'
for i in range(0, 38279):
    if (dfEdit.Description[i] != 'HOMICIDE'): 
        dfEdit = dfEdit.drop(i)

# View the values of the 'Description' column
dfEdit['Description'].unique()

# Write code to inspect the data frame
dfEdit

# Drop rows with null values
dfEdit = dfEdit.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Write code to inspect the data frame
dfEdit

# Create dataframe to hold only numerical values
dfNum = dfEdit
dfNum = dfNum.drop(columns=['Description'])

# Write code to inspect the data frame
dfNum

# Generate a scatter plot of Description and District
plt.scatter(dfNum['Longitude'], dfNum['Latitude'])

# Sets background image of plot
img = plt.imread('UnsupervisedLearning/BaltimoreCrime2/BaltimoreMap.png')
fig, ax = plt.subplots()
ax.imshow(img, extent=[-76.72,-76.525,39.20,39.40])
ax.scatter(dfNum['Longitude'], dfNum['Latitude'])
plt.show()

# Define the MinMaxScaler object and save in a variable called scaler
scaler = MinMaxScaler()

# Scale the Frequency feature
scaler.fit(dfNum[['Longitude']])
dfNum['Longitude'] = scaler.transform(dfNum[['Longitude']])

# Scale the Neighborhood feature
scaler.fit(dfNum[['Latitude']])
dfNum['Latitude'] = scaler.transform(dfNum[['Latitude']])

#TODO: Write code to inspect the scaled values of age and income.
# Notice that all the values are between 0 and 1
dfNum

# Define a range for possible k values. In this example, we choose 1 to 10
k_rng = range(1,10)

# Declare an array to store the values from the sum of squared error values.
sse = []

# Using a for loop, go through each value in the k range and compute the sse value
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(dfNum[['Longitude','Latitude']])
    sse.append(km.inertia_)

# Plot the sse value for each k.
# Notice that 3 is the elbow value.
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

# Define a KMeans object with 3 as the value of k. Call the object km
km = KMeans(n_clusters=4)

# Write code to inspect km to see all the default parameters that we didn't have to specify
km #debug

# Save the prediction in an array and call it y_predicted
y_predicted = km.fit_predict(dfNum[['Longitude','Latitude']])

# Visualize the array. 
# Notice that every element is assigned one of the three cluster IDs: 0, 1, or 2
y_predicted

# Add a column to the data frame to store the predicted cluster ID of each element
dfNum['cluster'] = y_predicted
# Write code to inspect "homicideFreq"
dfNum

# Define the three data frames, df1, df2, df3, each belonging to one of the three clusters
clusterDf1 = dfNum[dfNum.cluster==0]
clusterDf2 = dfNum[dfNum.cluster==1]
clusterDf3 = dfNum[dfNum.cluster==2]
clusterDf4 = dfNum[dfNum.cluster==3]

# Plot the data frames with different colors to differentiate them
plt.scatter(clusterDf1.Longitude,clusterDf1['Latitude'],color='green')
plt.scatter(clusterDf2.Longitude,clusterDf2['Latitude'],color='red')
plt.scatter(clusterDf3.Longitude,clusterDf3['Latitude'],color='blue')
plt.scatter(clusterDf4.Longitude,clusterDf4['Latitude'],color='black')


# Define the labels on the x and y axes
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Plot the legend
plt.legend()

# Sets background image of clustered plot
img = plt.imread('UnsupervisedLearning/BaltimoreCrime2/BaltimoreMap.png')
fig, ax = plt.subplots()
ax.imshow(img, extent=[-.1,1.1,-.1,1.1])
ax.scatter(clusterDf1.Longitude,clusterDf1['Latitude'],color='green')
ax.scatter(clusterDf2.Longitude,clusterDf2['Latitude'],color='red')
ax.scatter(clusterDf3.Longitude,clusterDf3['Latitude'],color='blue')
ax.scatter(clusterDf4.Longitude,clusterDf4['Latitude'],color='black')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

