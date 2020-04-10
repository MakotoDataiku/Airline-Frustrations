import dataiku                               # Access to Dataiku datasets
import pandas as pd, numpy as np             # Data manipulation
from sklearn.decomposition import PCA        # The main algorithm
from matplotlib import pyplot as plt         # Graphing
import seaborn as sns                        # Graphing
from collections import defaultdict, Counter # Utils
sns.set(style="white")                       # Tuning the style of charts
import warnings                              # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.preprocessing import StandardScaler


mydataset = dataiku.Dataset("word_vectors_prep")
df = mydataset.get_dataframe()

companies = df.product_id.unique()
df_PCA = pd.DataFrame()
for company in companies:
    print(company)
    df_sub = df[df.product_id == company]
    print(df_sub.shape)
    # print(df_sub.noun_lemmatized)
    columns_to_drop = ["product_id", "noun_lemmatized", "k_means_clusters", "count"]
    X = df_sub.drop(columns_to_drop, axis =1)
    Y = df_sub[columns_to_drop]
    print("ok for now")

    ss = StandardScaler().fit(X)
    X_std = ss.transform(X)
    pca = PCA(n_components=2)
    result = pca.fit_transform(X_std)
    result_df = pd.DataFrame(result, columns=['PC_1','PC_2'], index=Y.index)
    df_PCA_sub = Y.join(result_df, how="inner")

    df_PCA = pd.concat([df_PCA, df_PCA_sub])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_PCA.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
py_recipe_output = dataiku.Dataset("word_vectors_PCA")
py_recipe_output.write_with_schema(df_PCA)

"""
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pyplot.scatter(result[:, 0], result[:, 1])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Here the PCA is a full SVD (k=d, we have not yet applied any "reduction").

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Display of the explained variance of the eigenvectors. <a id="explained-variance" />
# 
# The first thing to do after fitting a PCA algorihtm is to plot the **explained variance** of each eigenvector (how much information from the original data does each vector contain).
# 
# We also compute how many of these vectors (in order) must be used to retain 90% of the variance of the original dataset (you can change that figure below)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
VARIANCE_TO_KEEP = 0.9

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
plt.bar(range(sklearn_pca.n_components_), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(sklearn_pca.n_components_), [sklearn_pca.explained_variance_ratio_[:y].sum() for y in range(1,sklearn_pca.n_components_+1)], alpha=0.5, where='mid',label='cumulative explained variance')
plt.axhline(y=0.95, linewidth=2, color = 'r')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xlim([0, sklearn_pca.n_components_])
plt.legend(loc='best')
plt.tight_layout()

keep_recommend = [sklearn_pca.explained_variance_ratio_[:y].sum()>VARIANCE_TO_KEEP for y in range(1,sklearn_pca.n_components_+1)].count(False)
print "Number of components to keep to retain %s%% of the variance:" % (100*VARIANCE_TO_KEEP), keep_recommend, "out of the original", sklearn_pca.n_components_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Retaining only some vectors <a id="final-pca" />

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# You should decide now how many components you want to keep and change the following parameter.
# 
# By default we keep the recommended value from the above figure

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
retained_components_number = keep_recommend

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Let's run the PCA again but with a limited number of components this time

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sklearn_pca_final = PCA(n_components=retained_components_number)
Y_sklearn_final = sklearn_pca_final.fit_transform(X_std)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Visualizing the eigenvectors in the original feature space <a id="original-space" />

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Decomposition heatmap

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Each of our eigenvectors has a linear decomposition in the original feature space.
# 
# To understand which features were the most important, we can see how our eigenvectors are made of each original feature.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# For display reasons, we don't show all components if more than 50 (same for input variables)
n_components_to_show = min(50, sklearn_pca_final.n_components_)
n_input_features = sklearn_pca_final.components_.shape[1]

decomp_df = pd.DataFrame(sklearn_pca_final.components_[0:n_components_to_show],
                            columns=df.columns)
if decomp_df.shape[1] > 50:
    decomp_df = decomp_df[decomp_df.columns[0:50]]

fig = plt.figure(figsize=(n_input_features, n_components_to_show))
sns.set(font_scale=3)
sns.heatmap(decomp_df, square=True)
sns.set(font_scale=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Visualizing projected vectors

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# The final visualization that we can build is the visualization of both the original dataset and the transformed dataset,
# in the original feature space.
# 
# We are going to select two features of the original dataset, and show on a XY graph:
# 
# * A scatterplot of the original dataset
# * A scatterplot of the reduced dataset (after losing the unexplained varaince)
# * The projection of the first two eigenvectors on the two selected features.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if len(numerical_columns) >= 2:
    feat1 = numerical_columns[0]
    feat2 = numerical_columns[1]
else:
    raise ValueError("Failed to automatically select proper variables to plot, please select manually")

print "Will plot on these two features: '%s' and '%s'" % (feat1, feat2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Uncomment this to take control on the two variables
# feat1 = "my_feat1"
# feat2 = "my_feat2"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
idx_feat_1 = list(df.columns).index(feat1)
idx_feat_2 = list(df.columns).index(feat2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
samp=1000
X_std_samp = np.random.choice(X_std.shape[0], samp)
plt.plot(X_std[X_std_samp, idx_feat_1], X_std[X_std_samp , idx_feat_2], 'o', alpha=0.1)
colors = ["green", "red"]
i = 0
for length, vector in zip(sklearn_pca_final.explained_variance_ratio_, sklearn_pca_final.components_)[0:2]:
    i = i+1
    i = i % len(colors)
    v = vector * 50 * length
    plt.plot([0, v[idx_feat_1]], [0, v[idx_feat_2]], '-k', lw=3, color=colors[i], label='PCA eigenvector ' + str(i))
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title('Projection of the first two eigenvectors of the PCA on the rescaled space (' + feat1 + ' / ' + feat2 + ')')
plt.legend(loc='upper right')
plt.axis("equal")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
plt.plot(X[X_std_samp, idx_feat_1], X[X_std_samp , idx_feat_2], 'o', alpha=0.1)
colors = ["green", "red"]
i = 0
for length, vector in zip(sklearn_pca_final.explained_variance_ratio_, sklearn_pca_final.components_)[0:2]:
    i = i+1
    i = i % len(colors)
    #print vector
    v = ss.inverse_transform(vector * length * 50)
    #print v
    plt.plot([ss.mean_[idx_feat_1], v[idx_feat_1]], [ss.mean_[idx_feat_2], v[idx_feat_2]], '-k', lw=3, color=colors[i], label='PCA eigenvector ' + str(i))
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title('Projection of the first two eigenvectors of the PCA on the original space (' + feat1 + ' / ' + feat2 + ')')
plt.legend(loc='upper right')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
samp=1000
X_std_samp = np.random.choice(X_std.shape[0], samp)
X_new = sklearn_pca_final.inverse_transform(Y_sklearn_final)
plt.plot(X_std[X_std_samp, idx_feat_1], X_std[X_std_samp, idx_feat_2], 'o', alpha=0.2, color="blue", label="Rescaled original data")
plt.plot(X_new[X_std_samp, idx_feat_1], X_new[X_std_samp, idx_feat_2], 'ob', alpha=0.5, color="red", label="Inverse transform after PCA")
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title('Drift of sample values due to the loss of variance after PCA')
plt.legend(loc='upper right')
plt.axis("equal")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Applying the projection <a id="apply" />

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Finally, we probably want to actually apply the PCA on the original data, which gives us the projected dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_PCA = pd.DataFrame(Y_sklearn_final, columns=[("PCA_component_" + str(comp)) for comp in range(sklearn_pca_final.n_components)])

# Inserts back the date columns in the dataFrame with PCA applied
for date_col_idx in range(len(date_columns)):
    col = date_columns[date_col_idx]
    df_PCA.insert(date_col_idx , col, df_orig[col])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Uncomment to display the head of the transformed matrix
#df_PCA.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
word_vectors_pca = dataiku.Dataset("word_vectors_PCA")
word_vectors_pca.write_with_schema(pandas_dataframe)
"""