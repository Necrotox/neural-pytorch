import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('gt_2015.csv')

tsne = TSNE(n_components=2, perplexity=50, n_jobs=-1).fit(df)
data = pd.DataFrame(tsne.fit_transform(df))
sns.scatterplot(data[0], data[1])
plt.show()

sts = StandardScaler()
df_tr = pd.DataFrame(sts.fit_transform(df), columns=df.columns)

relust = []
for i in np.arange(0.5, 1.5,0.05):
    for j in np.arange(6,17,1):
        model = DBSCAN(eps=i, min_samples=j).fit(df_tr)
        df['points'] = model.labels_
        result_row = np.array([i,j,
                              len(df[df['points']==-1]),
                               len([i for i in df['points'].unique() if i != -1]),
                               calinski_harabasz_score(df[[i for i in df.columns if i not in ['points']]], model.labels_),
                               silhouette_score(df[[i for i in df.columns if i not in ['points']]], model.labels_),
                              ])
        relust.append(result_row)

data_d = pd.DataFrame(relust, columns=['eps', 'min_samples', 'кол-во выбросов', 'кол-во кластеров', 'calinski_harabasz_score','silhouette_score'])
data_d

dbs = DBSCAN(eps=1, min_samples=11).fit(df_tr)
df['points'] = dbs.labels_

columns = [i for i in df.columns if i not in ['points']]
df_clu = df.groupby('points').agg({x: ['mean', 'std'] for x in columns}).T

func = ['mean', 'std']

glob = np.array([[df[col].apply(i) for i in func] for col in columns]).flatten()
df_rel = df_clu.copy()
df_rel['global'] = glob
cluster = np.arange(-1,30,1)

for label in cluster:
    df_rel[label] = 100*(df_rel[label] / df_rel['global']) - 100

sns.heatmap(data=df_rel, cmap='coolwarm')