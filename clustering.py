# -*- coding: utf-8 -*-
"""
Program to run feature engineering and perform k-means and DBSCAN clustering over PM2.5 concentration data

@author: alima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())

###############################################################################
### Step 1: Feature engineering/selection and paraneter/function initiation ###
###############################################################################

df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\pm_source_clustering_ml_master.xlsx'))
cols_keep = ['PM2.5_smooth', 'Der C', 'PM2.5_OD', 'Proximity to Baseline', 'Nearest Peak Proximity', 'I/O', 'wdir', 'wspd', 'Mode']
df = df[cols_keep]
df.dropna(inplace = True)



plt.figure(figsize=(8, 6))
plt.scatter(df['PM2.5_smooth'], df['Der C'], color = 'blue', alpha = 0.7, edgecolor = 'k')

# Labels and title
plt.xlabel('X-axis Variable')
plt.ylabel('Y-axis Variable')
plt.title('Two-Way Scatter Plot')

# Show grid for readability
plt.grid(True)
plt.show()


## Correlational and scatter plot matrix
corr_matrix = df.corr()
print(corr_matrix)

## Plotting the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.title('Correlation Matrix of Features')
plt.show()

## Plotting the scatter matrix
scatter_matrix(df, figsize = (10, 10), diagonal = 'kde', alpha=0.7)
plt.suptitle('Scatter Matrix of Features', y = 1.02)  # Add a title
# Set axis labels and title with custom font sizes
plt.xticks(fontsize = 4)
plt.yticks(fontsize = 4)
plt.savefig(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\plots\correlational_matrix.jpg', format = 'jpg', dpi = 1600, bbox_inches = 'tight')
plt.show()


## Centroid selections: 3 centeriods initially
n_clusters = 3
step = 3000 # for time-series data splitting in the plotting

## Centeroid determination functions  
def quantile_initialization(data, n_clusters): # centeroid selection based on qunatile
    data = np.array(data)
    n_features = data.shape[1]
    quantiles = np.linspace(0, 1, n_clusters + 2)[1:-1]  # Avoid 0 and 1 quantiles
    centroids = np.zeros((n_clusters, n_features))
    for feature in range(n_features):
        centroids[:, feature] = np.quantile(data[:, feature], quantiles)
    return centroids


def mode_initialization(data, n_clusters): # centeroid selection based on distribution modes
    # Convert data to numpy array if needed
    data = np.array(data)
    n_features = data.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    
    for feature in range(n_features):
        # Perform kernel density estimation
        kde = gaussian_kde(data[:, feature], bw_method = 'scott')
        
        # Create a range of values over which to evaluate the KDE
        feature_range = np.linspace(data[:, feature].min(), data[:, feature].max(), 1000)
        kde_values = kde(feature_range)
        
        # Find peaks in the KDE which correspond to the modes of the distribution
        peaks, _ = find_peaks(kde_values)
        
        # Sort peaks by their KDE value to get the highest modes
        mode_indices = peaks[np.argsort(kde_values[peaks])[-n_clusters:]]
        mode_values = feature_range[mode_indices]
        
        # If fewer modes are found than n_clusters, fill remaining centroids randomly within the feature range
        if len(mode_values) < n_clusters:
            additional_modes = np.random.uniform(data[:, feature].min(), data[:, feature].max(), n_clusters - len(mode_values))
            mode_values = np.concatenate([mode_values, additional_modes])
        
        # Assign modes as centroids for the current feature
        centroids[:, feature] = mode_values
    
    return centroids

def graph_plotter(df): # Only when you have PM2.5 smooth data
    for n in range(int(len(df)/step)):
        df_select = df.loc[n*step : (n+1)*step + 1 ,['PM2.5_smooth', 'cluster']]
        plt.scatter(df_select[df_select['cluster'] == 'source'].index, df_select[df_select['cluster'] == 'source']['PM2.5_smooth'], 
                    color = 'r', s = 2)
        plt.scatter(df_select[df_select['cluster'] == 'decay'].index, df_select[df_select['cluster'] == 'decay']['PM2.5_smooth'],
                    color = 'b', s = 0.3)
        plt.scatter(df_select[df_select['cluster'] == 'base'].index, df_select[df_select['cluster'] == 'base']['PM2.5_smooth'], 
                    color = 'g', s = 0.1)
        
        plt.xlabel('Time (Min) From the Start of the Study')
        
        plt.ylim(0, 100)
        plt.yticks(range(0, 101, 20))
        plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
        
        custom_lines = [
            Line2D([0], [0], color = 'r', lw = 2),    
            Line2D([0], [0], color = 'b', lw = 2),  
            Line2D([0], [0], color = 'g', lw = 2)]
            
        legend = plt.legend(custom_lines, ['Source', 'Decay', 'Base'], loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
        legend.get_frame().set_edgecolor("black")  
        plt.show()

def graph_plotter_log(df): 
    for n in range(int(len(df)/step)):
        df_select = df.loc[n*step : (n+1)*step + 1 ,['PM2.5_smooth', 'cluster']]
        plt.scatter(df_select[df_select['cluster'] == 'source'].index, df_select[df_select['cluster'] == 'source']['PM2.5_smooth'], 
                    color = 'r', s = 2)
        plt.scatter(df_select[df_select['cluster'] == 'decay'].index, df_select[df_select['cluster'] == 'decay']['PM2.5_smooth'],
                    color = 'b', s = 0.3)
        plt.scatter(df_select[df_select['cluster'] == 'base'].index, df_select[df_select['cluster'] == 'base']['PM2.5_smooth'], 
                    color = 'g', s = 0.1)
        
        plt.xlabel('Time (Min) From the Start of the Study')
        
        plt.ylim(1, 200)
        plt.yscale('log')
        plt.yticks(np.logspace(0, 2, 3))
        plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
        
        custom_lines = [
            Line2D([0], [0], color = 'r', lw = 2),    
            Line2D([0], [0], color = 'b', lw = 2),  
            Line2D([0], [0], color = 'g', lw = 2)]
            
        legend = plt.legend(custom_lines, ['Source', 'Decay', 'Base'], loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
        legend.get_frame().set_edgecolor("black")  
        plt.show()


#####################################################################################
### Step 2: k-means without dimensionality reduction with quantile_initialization ###
#####################################################################################

## k-means with quantile-determined centeroids
initial_centroids_quantile = quantile_initialization(df.to_numpy(), n_clusters)
kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids_quantile, random_state = 42, max_iter = 10000)
kmeans.fit(df)
labels = kmeans.predict(df)
df['cluster'] = labels
clusters = dict(zip(np.argsort(initial_centroids_quantile[:, 0]) ,  ['base', 'source', 'decay']))
df['cluster'].replace(clusters, inplace = True)

## Plotting the graphs
graph_plotter(df)


#################################################################################
### Step 3: k-means without dimensionality reduction with mode_initialization ###
#################################################################################

df.drop('cluster', axis = 1, inplace = True, errors = 'ignore') # dropping clusters from the previous modeling (to recreate it)

## Centeroid determination based on distribution modes and Kernel Density Estimation (KDE)
initial_centroids_kde = mode_initialization(df.to_numpy(), n_clusters)
kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids_kde, random_state = 42, max_iter = 10000)
kmeans.fit(df)
labels = kmeans.predict(df)
df['cluster'] = labels
clusters = dict(zip(np.argsort(initial_centroids_quantile[:, 0]) ,  ['base', 'source', 'decay']))
df['cluster'].replace(clusters, inplace = True)


graph_plotter(df)


##################################################################################
### Step 4: k-means without dimensionality reduction with pre-defined clusters ###
##################################################################################

df.drop('cluster', axis = 1, inplace = True, errors = 'ignore')
initial_centroids = pd.DataFrame({'PM2.5_smooth': [20, 30, 10],
                                    'Der C': [50, -10, 0],
                                    'PM2.5_OD': [5, 5, 5],
                                    'Proximity to Baseline': [10, 10, 0],
                                    'Nearest Outlier Proximity': [5, 5, 20],
                                    'I/O': [2, 2.5, 1],
                                    'wdir': [300, 300, 300],
                                    'wspd': [10, 10, 10],
                                    'Mode': [0, 0, 0]})

## Centeroid determination based on pre-difned (i.e., user-defined values)
kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids, random_state = 42, max_iter = 10000)
kmeans.fit(df)
labels = kmeans.predict(df)
df['cluster'] = labels
clusters = dict(zip(np.argsort(initial_centroids_quantile[:, 0]) ,  ['base', 'source', 'decay']))
df['cluster'].replace(clusters, inplace = True)

graph_plotter(df)


#########################################################
### Step 5: k-means with PCA dimensionality reduction ###
#########################################################

df.drop('cluster', axis = 1, inplace = True, errors = 'ignore')

## Standardizing the data (important for PCA)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

## Performing PCA for all components (9 in this case) for Elbom method and finding the optimized No. Components
pca = PCA(n_components = 9)
pca.fit(df_scaled)
explained_variance = pca.explained_variance_ratio_
# Cumulative explained variance
# cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize = (8, 5))
plt.plot(range(1, 10), explained_variance, marker = 'o', linestyle = '-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Elbow Method for PCA')
plt.grid(True)
plt.show() # figure shows 3 components are best

## Performging PCA with 3 compoentns 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA(n_components = 3)
pca_df = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data = pca_df, columns = [f'PC{i+1}' for i in range(pca_df.shape[1])])

## quantile_initialization for centeroid allocation
initial_centroids_quantile = quantile_initialization(pca_df.to_numpy(), n_clusters)
kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids_quantile, random_state = 42, max_iter = 10000)
kmeans.fit(pca_df)
labels = kmeans.predict(pca_df)
pca_df['cluster'] = labels
pca_df['cluster'].replace(clusters, inplace = True)
pca_df_qua = pd.concat([df, pca_df['cluster']], axis = 1)

graph_plotter(pca_df_qua)

## mode_initialization for centeroid allocation

pca_df.drop('cluster', axis = 1, inplace = True, errors = 'ignore')

initial_centroids_kde = mode_initialization(pca_df.to_numpy(), n_clusters)
kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids_kde, random_state = 42, max_iter = 10000)
kmeans.fit(pca_df)
labels = kmeans.predict(pca_df)
pca_df['cluster'] = labels
pca_df['cluster'].replace(clusters, inplace = True)
pca_df_kde = pd.concat([df, pca_df['cluster']], axis = 1)

graph_plotter(pca_df_kde)


###########################################################
### Step 6: k-means with t-SNE dimensionality reduction ###
###########################################################

df.drop('cluster', axis = 1, inplace = True, errors = 'ignore')

## Applying t-SNE with two components
tsne = TSNE(n_components = 2, random_state = 42, perplexity = 30, n_iter = 10000)
tsne_df = tsne.fit_transform(df_scaled)
tsne_df = pd.DataFrame(data = tsne_df, columns = ['tSNE1', 'tSNE2'])

# Plot the t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_df['tSNE1'], tsne_df['tSNE2'], s=50, alpha=0.7)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.grid(True)
plt.show()


## Apply k-means clustering on the t-SNE output
n_clusters = 3  # Set the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state = 42)
kmeans_labels = kmeans.fit_predict(tsne_df)

# Create a DataFrame with t-SNE results and cluster labels
tsne_df = pd.DataFrame(data = tsne_df, columns = ['tSNE1', 'tSNE2'])
tsne_df['cluster'] = kmeans_labels

tsne_df['cluster'].replace(clusters, inplace = True)

tsne_df = pd.concat([df, tsne_df['cluster']], axis = 1)
tsne_df.dropna(inplace = True)
tsne_df.to_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\tsne_two_comp.xlsx'), index = False) # Saving t-SNE results as t-SNE runtime is really long

graph_plotter(pca_df_kde)


#####################################################################################
### Step 6: k-means with no dimensionality reduction and manual feature selection ###
#####################################################################################

## Changing No. clusters from 3 to 5 to include both mild  and extensive source and decays
clusters = {0: 'source', # extensive sources 
            1: 'source', # mild source
            2: 'decay', # extensive decay
            3: 'decay', # mild decay
            4: 'base'}


n_clusters = len(clusters.keys())
df.drop('cluster', axis = 1, inplace = True, errors = 'ignore')
initial_centroids_kde = mode_initialization(df['Der C'].to_numpy().reshape(len(df),1), n_clusters)
initial_centroids_quantile = quantile_initialization(df['Der C'].to_numpy().reshape(len(df),1), n_clusters)

## Except C derivative, other features are removed from the model as derivative of C is assumed to be more important
initial_centroids_part = pd.DataFrame({# 'PM2.5_smooth': [80, 60, 60, 40, 5],
                                       'Der C': [2, 0.25, -1, -0.5, 0]
                                       # 'PM2.5_OD': [5, 3.2, 5.5],
                                       # 'Proximity to Baseline': [0, 50, 40],
                                       # 'Nearest Outlier Proximity': [20, 10, 60]
                                       # 'I/O': [20, 15, 0.8]
                                       })

kmeans = KMeans(n_clusters = n_clusters, init = initial_centroids_part, random_state = 42, max_iter = 1000)
kmeans.fit(pd.DataFrame(df['Der C']))
labels = kmeans.predict(pd.DataFrame(df['Der C']))
df['cluster'] = labels
df['cluster'].replace(clusters, inplace = True)

graph_plotter(df)




## Plotting the derivative of C 
series = df['Der C'] # + abs(df['Der C'].min())

plt.figure(figsize = (8, 6))
sns.histplot(series, kde = True, bins = 6, color='skyblue')
plt.xlabel('Der C')
# plt.xscale('log')  # Set x-axis to logarithmic scale

plt.ylabel('Frequency')
plt.title('Distribution of Der C')
plt.grid(True)
plt.show()


print(len(df[(df['cluster'] == 'source') | (df['cluster'] == 'mild source')]))
print((len(df[(df['cluster'] == 'source') | (df['cluster'] == 'mild source')])/len(df)) * 100)



####################################################################################
### Step 7: DBSCAN with no dimensionality reduction and manual feature selection ###
####################################################################################

## Building DBSCAN model
dbscan = DBSCAN(eps = 0.2, min_samples = 200) # 1500 is selected as it captures more of source regime periods
dbscan_labels = dbscan.fit_predict(pd.DataFrame(df['PM2.5_smooth']))
df['cluster'] = dbscan_labels


plt.figure(figsize = (8, 6))
plt.scatter(df[df['cluster'] == -1]['PM2.5_smooth'], df[df['cluster'] == -1]['Der C'], 
            color = 'r', alpha = 0.7, edgecolor = 'k')

plt.scatter(df[df['cluster'] == 0]['PM2.5_smooth'], df[df['cluster'] == 0]['Der C'], 
            color = 'b', alpha = 0.7, edgecolor = 'k')

plt.scatter(df[df['cluster'] == 1]['PM2.5_smooth'], df[df['cluster'] == 1]['Der C'], 
            color = 'g', alpha = 0.7, edgecolor = 'k')

plt.xlabel('X-axis Variable')
plt.ylabel('Y-axis Variable')
plt.title('Two-Way Scatter Plot')

plt.grid(True)
plt.show()

## Correction to cluster labelling ()
df['cluster'] = np.where((df['Der C'] > 0) & (df['cluster'] == -1), 'source', 'non-source')

## Plotting the source vs. non-source curves for all periods
for n in range(int(len(df)/step)):
    df_select = df.loc[n*step : (n+1)*step + 1 ,['PM2.5_smooth', 'cluster']]
    plt.scatter(df_select[df_select['cluster'] == 'source'].index, df_select[df_select['cluster'] == 'source']['PM2.5_smooth'], 
                color = 'r', s = 2)
    plt.scatter(df_select[df_select['cluster'] != 'source'].index, df_select[df_select['cluster'] != 'source']['PM2.5_smooth'],
                color = 'b', s = 0.3)
    
    plt.xlabel('Time (Min) From the Start of the Study')
    
    plt.ylim(0, 140)
    plt.yticks(range(0, 141, 20))
    plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
    
    custom_lines = [
        Line2D([0], [0], color = 'r', lw = 2),    
        Line2D([0], [0], color = 'b', lw = 2)]  
                
    legend = plt.legend(custom_lines, ['Source', 'Non-source'], loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
    legend.get_frame().set_edgecolor("black")  
    legend.get_frame().set_edgecolor("black")  
    plt.show()



print(len(df[df['cluster'] == 'source']))
print((len(df[df['cluster'] == 'source'])/len(df)) * 100)


######################################################################
### Step 8: DBSCAN trial and error for finding the right min_samlt ###
######################################################################

## for loop with min_sample varying from 100 through 2000
for i in range(100, 2001, 100):
    dbscan = DBSCAN(eps = 0.2, min_samples = i)  # Adjust eps and min_samples as needed
    dbscan_labels = dbscan.fit_predict(pd.DataFrame(df['PM2.5_smooth']))
    df['cluster'] = dbscan_labels
    
    # df_select only for the first 3000 minutes
    df_select = df.loc[0 : 3000 ,['PM2.5_smooth', 'cluster']]
    plt.scatter(df_select[df_select['cluster'] == -1].index, df_select[df_select['cluster'] == -1]['PM2.5_smooth'], 
                color = 'r', s = 2)
    plt.scatter(df_select[df_select['cluster'] != -1].index, df_select[df_select['cluster'] != -1]['PM2.5_smooth'],
                color = 'b', s = 0.3)
    
    plt.xlabel('Time (Min) From the Start of the Study')
    
    plt.ylim(0, 140)
    custom_lines = [
        Line2D([0], [0], color = 'r', lw = 2),    
        Line2D([0], [0], color = 'b', lw = 2),  
        Line2D([0], [0], color = 'g', lw = 2)]
        
    legend = plt.legend(custom_lines, ['Source', 'Decay', 'Base'], loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
    legend.get_frame().set_edgecolor("black")  
    plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
    
    
    legend.get_frame().set_edgecolor("black")  
    plt.show()

