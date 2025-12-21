# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
- Collect Data
- Gather customer information (e.g., age, income, spending score, purchase history).
- Clean the data (remove missing values, normalize if needed).
- Choose Number of Clusters (k)
- Decide how many customer groups you want (e.g., 3 segments: low spenders, medium spenders, high spenders).
- Often chosen using the "Elbow Method" (plotting cost vs. k).
- Initialize Centroids
- Randomly pick k points from the dataset as the starting "centers" of clusters.
- Assign Customers to Nearest Cluster
- For each customer, calculate the distance (usually Euclidean distance) to each centroid.
- Assign the customer to the cluster with the closest centroid.
- Update Centroids
- After assignment, recalculate the centroid of each cluster:
- New centroid = average of all points in that cluster.
- Repeat Steps 4 & 5
- Keep reassigning customers and updating centroids until:
- Centroids stop changing significantly, OR
- A maximum number of iterations is reached.
- Result: Customer Segments
- Each cluster now represents a customer segment (e.g., budget shoppers, loyal premium buyers, occasional spenders).
- Use these segments for marketing strategies, personalized offers, or customer analysis.


## Program:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)

Output:
![K Means Clustering for Customer Segmentation](sam.png)
<img width="955" height="710" alt="image" src="https://github.com/user-attachments/assets/679e930f-de63-4688-bb74-5cd10e141837" />
<img width="898" height="602" alt="image" src="https://github.com/user-attachments/assets/4a2f32a5-e046-4e65-a8aa-029737b5a02f" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
