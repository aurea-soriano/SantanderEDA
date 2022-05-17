import pandas as pd
import numpy as np
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
sns.set(style="white", color_codes=True)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.iloc[:,:-1]
y_train = train['TARGET']

X_test = test

print(train.head())
print(X_train.dtypes)
print(X_train.dtypes.value_counts())
print(X_train.describe())
print(X_train.select_dtypes(include=['int64']).nunique())

#identificando atributos com valores constantes
constant_features = X_train.nunique()
constant_features = constant_features.loc[constant_features.values==1].index

# remover essas colunas
X_train = X_train.drop(constant_features,axis=1)
X_test = X_test.drop(constant_features,axis=1)

# o ID nao adiciona informacao
X_train = X_train.drop("ID",axis=1)
X_test = X_test.drop("ID",axis=1)


print(X_train.isnull().values.any())


print(len(X_train.columns[X_train.isna().sum() > 0]))


cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]


# Estratégia 1 Remover colunas
reduced_train = X_train.drop(cols_with_missing, axis=1)
reduced_test = X_test.drop(cols_with_missing[:-1], axis=1)

print(reduced_train.isnull().values.any())



cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]


# Estratégia 2 usamos SimpleImputer para substituir valores ausentes pelo valor médio ao longo de cada coluna.
from sklearn.impute import SimpleImputer

# Imputation
imputer = SimpleImputer()
imputed_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_test = pd.DataFrame(imputer.fit_transform(X_test))

# Imputação removeu nomes de colunas; coloque-os de volta
imputed_train.columns = X_train.columns
imputed_test.columns = X_test.columns


print(imputed_train.isnull().values.any())


X_train = imputed_train
X_test = imputed_test


print(X_train.describe().loc[['min','max']])

data = X_train[X_train.columns[6]].values

f = Fitter(data,distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
           #distributions= get_common_distributions())
f.fit()
f.summary






# normalizando
X_train_norm = preprocessing.normalize(X_train)


top=10
plot_data = pd.DataFrame(X_train.nunique().sort_values(ascending=False)).reset_index().head(top)

plt.figure(figsize=(10,10))
plt.barh(plot_data['index'][::-1], plot_data[0][::-1])
plt.show()



plt.figure(figsize = (20,25))
sns.heatmap(X_train.corr())
plt.show()


data = X_train_norm[:,1]
sns.distplot(data, hist=True, kde=True,
             bins=10, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()


sns.distplot(data, hist=True, kde=True,
             bins=30, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()


sns.distplot(data, hist=True, kde=True,
             bins=60, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()



# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(data=X_train, x="var3", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=X_train, x="var15", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=X_train, x="saldo_medio_var33_hace3", kde=True, color="gold", ax=axs[1, 0])
sns.histplot(data=X_train, x="var38", kde=True, color="teal", ax=axs[1, 1])

plt.show()


sns.boxplot( y=X_train["var15"], x=y_train)
plt.title("Distribution of var15")
plt.show()




sns.scatterplot( x= X_train["var38"], y=X_train["var15"], hue=y_train)
plt.xlabel("var38")
plt.ylabel("var15")
plt.show()


sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var4") \
   .add_legend()
plt.show()


sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "num_var4") \
   .add_legend()


train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=60

sns.pairplot(train[['var15','var36','var38','TARGET']], hue="TARGET", size=2, diag_kind="kde")


pca = PCA(n_components=X_train_norm.shape[1])
Xt = pca.fit_transform(X_train_norm)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y_train)
plt.show()





X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_train_norm)
plot = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
plt.show()



svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
X_tsvd= svd.fit_transform(X_train_norm)
plot = plt.scatter(X_tsvd[:,0], X_tsvd[:,1], c=y_train)
plt.show()


reducer = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
               learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
               spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False, # default False, Controls verbosity of logging.
               unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded.
              )

# Fit and transform the data
X_trans = reducer.fit_transform(X_train_norm)
plot = plt.scatter(X_trans[:,0], X_trans[:,1], c=y_train)
plt.show()
