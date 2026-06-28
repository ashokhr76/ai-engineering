from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())


## Include sepal length and width in the scatter plot
scatter = plt.scatter( df['petal length (cm)'], df['petal width (cm)'], c=df['target'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset - Petal Length vs Petal Width')
plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names)
plt.show()

minmax = MinMaxScaler()
minmax.fit(df[['petal length (cm)']])
df['petal length (cm)'] = minmax.transform(df[['petal length (cm)']])
minmax.fit(df[['petal width (cm)']])
df['petal width (cm)'] = minmax.transform(df[['petal width (cm)']])

kmeanCluster = KMeans(n_clusters=3)
y_predicted = kmeanCluster.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
df['cluster'] = y_predicted

print(df.head())

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='green')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='red')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='black')

plt.scatter(kmeanCluster.cluster_centers_[:,0], kmeanCluster.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()  

### How to identify the optimal number of clusters using the elbow method
sse = []
k_rng = range(1, 10)
for k in k_rng:
    kmeanCluster = KMeans(n_clusters=k)
    kmeanCluster.fit(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(kmeanCluster.inertia_)

plt.plot(k_rng, sse)
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()  
