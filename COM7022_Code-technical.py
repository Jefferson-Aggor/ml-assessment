#!/usr/bin/env python
# coding: utf-8

# # COM7022 Machine Learning Portfolio: code
# 
# Student ID: 25138486
# 
# The sections of this notebook follow the sections of the report in the same order, and each figure in Task 1 is a screenshot of the cell under the same section number. Run the notebook from top to bottom with **Sales-[5444].csv** in the same folder.

# ## Setup

# In[1]:


import re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score)

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
SEED = 42


# # Task 1
# ## 2. Data
# ### 2.1 Structure (Figure 1)

# In[2]:


df = pd.read_csv('Sales-[5444].csv', encoding='utf-8-sig')
raw = df.copy()   # untouched copy, kept for before-and-after comparisons
print('Shape:', df.shape)
df.head(3)


# ### 2.2 Data types (Figure 2)

# In[3]:


print(df.dtypes)


# ### 2.3 Missing values (Figure 3)

# In[4]:


missing = df.isna().sum()
print(missing)
print('Total missing cells:', missing.sum(), f'({missing.sum() / df.size:.2%})')
print('Rows with at least one missing value:', df.isna().any(axis=1).sum())


# ### 2.4 Descriptive statistics (Figure 4)

# In[5]:


print(df.describe().round(2))

sold = df['sold'].dropna()
top10 = sold.sort_values(ascending=False).head(len(sold) // 10)
print('\nMedian units sold:', sold.median(), '| mean units sold:', round(sold.mean()))
print(f'Share of all units sold by the top 10% of listings: {top10.sum() / sold.sum():.0%}')


# ## 3. Methods
# ### 3.1 Data preparation
# #### 3.1.1 Column names and text gaps (Figure 5)

# In[6]:


df.columns = df.columns.str.lower()
df = df.rename(columns={'catatory': 'category'})
print(list(df.columns))

text_cols = ['item_name', 'category', 'text', 'pricewithcurrency']
print('Missing text values before:', df[text_cols].isna().sum().sum())
df[text_cols] = df[text_cols].fillna('Unknown')
print('Missing text values after: ', df[text_cols].isna().sum().sum())


# #### 3.1.2 Approach to missing numbers (Figure 6)

# In[7]:


# Do listings with a missing value sell differently? If so, deleting them would bias the results.
print('Median units sold, by whether the date is missing:')
print(raw.groupby(raw['lastUpdated'].isna())['sold'].median())

# The numeric columns are skewed, so the median is a fairer fill value than the mean.
print(raw[['price', 'sold', 'available']].agg(['skew', 'median', 'mean']).round(2))


# #### 3.1.3 Price and units sold (Figure 7)

# In[8]:


# Recover missing values from the text columns, then fill any gaps left with the median
price_text = df['pricewithcurrency'].str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0]
sold_text = df['text'].str.replace(',', '').str.extract(r'(\d+)\s+sold', flags=re.IGNORECASE)[0]

for col, from_text in [('price', price_text.astype(float)), ('sold', sold_text.astype(float))]:
    both = df[col].notna() & from_text.notna()
    match = (from_text[both] == df.loc[both, col]).mean()
    print(f'{col}: the text matches the stored value for {match:.1%} of listings')
    missing_before = df[col].isna().sum()
    df[col] = df[col].fillna(from_text)
    recovered = missing_before - df[col].isna().sum()
    df[col] = df[col].fillna(df[col].median())
    print(f'   missing: {missing_before} | recovered from the text: {recovered}',
          f'| filled with the median: {missing_before - recovered}')


# #### 3.1.4 Update date (Figure 8)

# In[9]:


# remove the time zone (e.g. 'PDT'), then convert the text to a date
date_text = df['lastupdated'].str.replace(r'\s+[A-Za-z]{2,4}$', '', regex=True)
df['lastupdated_dt'] = pd.to_datetime(date_text, format='%b %d, %Y %H:%M:%S', errors='coerce')
failed = df['lastupdated'].notna() & df['lastupdated_dt'].isna()
print('Missing dates:', df['lastupdated_dt'].isna().sum())
print('Text that failed to convert:', failed.sum())

df['lastupdated_dt'] = df['lastupdated_dt'].fillna(df['lastupdated_dt'].median())
ref_date = df['lastupdated_dt'].max() + pd.Timedelta(days=1)
df['days_since_update'] = (ref_date - df['lastupdated_dt']).dt.days
df = df.drop(columns=['lastupdated'])     # the text version is no longer needed
print('Reference date:', ref_date.date())
print(df['days_since_update'].describe().round(1))


# #### 3.1.5 Stock (Figures 9 and 10)

# In[10]:


# What does the text say about stock, and which of these listings have no stock number?
t = df['text']
conditions = [t.str.contains('More than 10 available', case=False),
              t.str.contains('Last One', case=False),
              t.str.contains('Limited quantity', case=False),
              t.str.contains(r'^\s*[\d,]+ available', regex=True)]
labels = ['More than 10 available', 'Last One', 'Limited quantity', 'Exact number']
status = pd.Series(np.select(conditions, labels, default='Other or no text'), index=df.index)
print(pd.crosstab(status.rename('stock status'), df['available'].isna().rename('stock missing'),
                  margins=True))
more_than_10 = status == 'More than 10 available'
print('\nValue stored for "More than 10 available":', df.loc[more_than_10, 'available'].unique())


# In[11]:


df['available_10plus'] = more_than_10.astype(int)     # true stock is only known to be at least 10
df['available_limited'] = (status == 'Limited quantity').astype(int)

last_one = df['available'].isna() & (status == 'Last One')
df.loc[last_one, 'available'] = 1
stock_imputed = df['available'].isna()        # remembered for the stock sensitivity test
print('"Last One" set to 1:', last_one.sum())
print('Filled with the median of', df['available'].median(), ':', stock_imputed.sum())
df['available'] = df['available'].fillna(df['available'].median())

print('\nMedian price and units sold, by "Limited quantity":')
print(df.groupby('available_limited')[['price', 'sold']].median())


# #### 3.1.6 Product category (Figure 11)

# In[12]:


print('Distinct labels:', df['category'].nunique(),
      '| labels used by fewer than five listings:', (df['category'].value_counts() < 5).sum())

c = df['category'].str.lower().str.replace('~', ' ')
c = c.str.replace(r'\s+', ' ', regex=True).str.strip()
has = lambda pattern: c.str.contains(pattern, regex=True)
rules = [   # checked in order: the first rule that matches decides the product type
    # mixed types, e.g. 'eau de toilette & parfum'
    ('other', (has('toil') & has('parf')) | (has('toil') & has('cologne'))
              | (has('parf') & has('cologne'))),
    # not a single fragrance: gift sets, deodorants, oils
    ('other', has(r'deodorant|\boil\b|roll on|freshener|body spray|splash|gift|\bset')),
    ('eau_de_toilette', has(r'toil|\bedt\b')),
    ('eau_de_parfum', has(r'eau de parf|eau de perf|\bedp\b')),
    ('eau_de_cologne', has(r'cologne|\bedc\b')),
    ('parfum', has(r'parfum|elixir|extrait')),
    ('perfume_generic', has(r'perfume|fragrance')),
]
df['category_grouped'] = np.select([m for _, m in rules], [name for name, _ in rules], default='other')
print(df['category_grouped'].value_counts())


# #### 3.1.7 Outliers (Figures 12 and 13)

# In[13]:


num = ['price', 'sold', 'available', 'days_since_update']
q1, q3 = df[num].quantile(0.25), df[num].quantile(0.75)
iqr = q3 - q1
print('Values outside 1.5 x IQR:')
print(((df[num] < q1 - 1.5 * iqr) | (df[num] > q3 + 1.5 * iqr)).sum())

print('\nLargest units sold, checked against the text:')
largest = df.nlargest(3, 'sold')[['details', 'sold', 'text']]
print(largest.assign(details=largest['details'].str[:30]).to_string())


# In[14]:


# Check every exact stock number in the text against the stored value
exact = df['text'].str.extract(r'^\s*([\d,]+) available')[0].str.replace(',', '').astype(float)
wrong = exact.notna() & (exact != df['available'])
print(df.loc[wrong, ['available', 'text']])
df.loc[wrong, 'available'] = exact[wrong]          # restore the lost thousands digit
print('Corrected:', wrong.sum())

print('\nMissing values left in the numeric features:', df[num].isna().sum().sum())
print(df[num].describe().round(2))


# ### 3.2 Feature selection (Figure 14)

# In[15]:


corr = df[num].corr(method='spearman')
plt.figure(figsize=(5.5, 4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Spearman correlation of the numeric features')
plt.tight_layout(); plt.show()


# ### 3.3 Transformation
# #### 3.3.1 Log transformation (Figure 15)

# In[16]:


df_log = np.log1p(df[num])          # log(1 + x), so zero stays defined
print(pd.DataFrame({'skew before': df[num].skew(), 'skew after': df_log.skew()}).round(2))

fig, axes = plt.subplots(2, 4, figsize=(14, 5))
for i, col in enumerate(num):
    axes[0, i].hist(df[col], bins=40); axes[0, i].set_title(f'{col} (original)')
    axes[1, i].hist(df_log[col], bins=40, color='darkorange'); axes[1, i].set_title(f'{col} (log)')
plt.tight_layout(); plt.show()


# #### 3.3.2 Scaling (Figure 16)

# In[17]:


scaled = pd.DataFrame(StandardScaler().fit_transform(df_log), columns=num)
summary = pd.DataFrame({'sd before': df_log.std(),
                        'mean after': scaled.mean(), 'sd after': scaled.std()})
print(summary.round(2) + 0.0)       # + 0.0 avoids printing -0.0


# #### 3.3.3 Encoding product type (Figure 17)

# In[18]:


cat_dummies = pd.get_dummies(df['category_grouped'], prefix='cat', dtype=int)
X = pd.concat([scaled, cat_dummies], axis=1)      # the final feature table
print('Feature table:', X.shape)
print(X.head(3).round(2).to_string())


# ### 3.4 Dimensionality reduction (PCA) (Figures 18 and 19)

# In[19]:


pca = PCA().fit(X)
ratio = pca.explained_variance_ratio_
print(pd.DataFrame({'variance': ratio, 'cumulative': ratio.cumsum()},
                   index=[f'PC{i + 1}' for i in range(len(ratio))]).round(3).head(5))
print('\nLoadings of the first four components:')
loadings = pd.DataFrame(pca.components_[:4].T, index=X.columns, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(loadings.round(2))


# In[20]:


pc = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(7, 4.5))
for ptype in sorted(df['category_grouped'].unique()):
    m = (df['category_grouped'] == ptype).values
    plt.scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.6, label=ptype)
plt.xlabel('PC1 (sales and stock)'); plt.ylabel('PC2 (days since update)')
plt.title('Listings on the first two principal components'); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()


# ### 3.5 Clustering methods and the number of clusters (Figures 20 to 22)

# In[21]:


def scores(X, labels):
    return {'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)}

rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    rows.append({'k': k, 'inertia': km.inertia_, **scores(X, km.labels_)})
k_scores = pd.DataFrame(rows).set_index('k').round(3)
print(k_scores)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
titles = ['Inertia (elbow)', 'Silhouette (higher is better)',
          'Davies-Bouldin (lower is better)', 'Calinski-Harabasz (higher is better)']
for ax, col, title in zip(axes, k_scores.columns, titles):
    ax.plot(k_scores.index, k_scores[col], marker='o'); ax.axvline(5, color='grey', linestyle='--')
    ax.set_title(title); ax.set_xlabel('k')
plt.tight_layout(); plt.show()


# In[22]:


from scipy.cluster.hierarchy import linkage, dendrogram

# Ward linkage joins the two groups whose merger adds the least within-cluster variance,
# the same quantity K-Means minimises, so the two methods are directly comparable
Z = linkage(X, method='ward')
steps = [f'{k + 1} -> {k} clusters' for k in range(6, 0, -1)]
print(pd.Series(Z[-6:, 2].round(1), index=steps, name='merge distance').to_string())
cut = (Z[-5, 2] + Z[-4, 2]) / 2          # a cut between these two merges leaves five clusters

plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode='lastp', p=30, no_labels=True, color_threshold=cut)
plt.axhline(cut, color='grey', linestyle='--', label='cut giving five clusters')
plt.ylabel('Merge distance'); plt.title('Ward dendrogram (last 30 merges)'); plt.legend()
plt.tight_layout(); plt.show()


# In[23]:


from sklearn.neighbors import NearestNeighbors

# DBSCAN needs eps, the distance within which listings count as neighbours.
# The k-distance plot sorts each listing's distance to its 10th nearest point (min_samples = 10);
# eps is read where the curve bends upwards.
dist = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)[0][:, -1]
dist = np.sort(dist)

# What DBSCAN produces at, below and above eps = 1.0
for eps in [0.75, 1.0, 1.25, 1.5]:
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    kept = lab != -1
    types = pd.crosstab(lab[kept], df.loc[kept, 'category_grouped']).gt(0).sum(axis=1)
    print(f'eps = {eps}: clusters = {len(set(lab[kept]))}, noise = {(~kept).mean():.0%},',
          f'clusters holding a single product type = {(types == 1).sum()}')

plt.figure(figsize=(7, 4))
plt.plot(dist); plt.axhline(1.0, color='grey', linestyle='--', label='eps = 1.0')
plt.xlabel('Listings sorted by distance'); plt.ylabel('Distance to 10th nearest point')
plt.title('k-distance plot for choosing eps'); plt.legend()
plt.tight_layout(); plt.show()


# ## 4. Results
# ### 4.1 Model comparison (Figures 23 and 24)

# In[24]:


models = {'K-Means (k = 5)': KMeans(n_clusters=5, n_init=20, random_state=SEED),
          'Hierarchical, Ward (k = 5)': AgglomerativeClustering(n_clusters=5, linkage='ward'),
          'Hierarchical, average (k = 5)': AgglomerativeClustering(n_clusters=5, linkage='average'),
          'DBSCAN (eps = 1.0, min_samples = 10)': DBSCAN(eps=1.0, min_samples=10)}
labels = {name: model.fit_predict(X) for name, model in models.items()}

rows = []
for name, lab in labels.items():
    kept = lab != -1       # DBSCAN marks noise as -1; scores use the clustered listings only
    rows.append({'model': name, 'clusters': len(set(lab[kept])),
                 'noise %': round((~kept).mean() * 100, 1),
                 'largest cluster %': round(np.bincount(lab[kept]).max() / len(lab) * 100, 1),
                 **scores(X[kept], lab[kept])})
print(pd.DataFrame(rows).set_index('model').round(3).to_string())

print('\nDBSCAN clusters by product type (-1 = noise):')
dbscan = labels['DBSCAN (eps = 1.0, min_samples = 10)']
print(pd.crosstab(dbscan, df['category_grouped'], rownames=['DBSCAN cluster']))


# In[25]:


# Does the selected model change with other reasonable choices? Agreement is measured with the
# adjusted Rand index (ARI): 1 = identical segments, 0 = no more agreement than chance
base = labels['K-Means (k = 5)']
X_flags = pd.concat([X, df[['available_limited', 'available_10plus']]], axis=1)   # flags added back
X_pca4 = PCA(n_components=4).fit_transform(X)                                      # 86.7% of variation
for name, data in [('Stock flags added back', X_flags), ('Four principal components', X_pca4)]:
    lab = KMeans(n_clusters=5, n_init=20, random_state=SEED).fit_predict(data)
    print(f'{name}: ARI with the selected model = {adjusted_rand_score(base, lab):.3f}')

runs = [KMeans(n_clusters=5, n_init=20, random_state=s).fit_predict(X) for s in range(10)]
ari = [adjusted_rand_score(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)]
print(f'Ten random seeds: mean ARI between runs = {np.mean(ari):.3f} (lowest {np.min(ari):.3f})')


# ### 4.2 The five segments (Figures 25 and 26)

# In[26]:


df['cluster'] = labels['K-Means (k = 5)']
df['est_sales_value'] = df['price'] * df['sold']   # estimated cumulative sales value, not revenue
SEGMENTS = ['Bestsellers', 'Bulk stockists', 'Premium slow movers',
            'Stale listings', 'Budget slow movers']

def name_segments(labels, data):
    # K-Means numbers its clusters arbitrarily, so each cluster is named from its medians
    med = data.groupby(labels)[num].median()
    names, remaining = {}, list(med.index)
    for col, name in [('available', 'Bulk stockists'), ('price', 'Premium slow movers'),
                      ('days_since_update', 'Stale listings'), ('sold', 'Bestsellers')]:
        top = med.loc[remaining, col].idxmax(); names[top] = name; remaining.remove(top)
    names[remaining[0]] = 'Budget slow movers'
    return pd.Series(labels, index=data.index).map(names)

df['segment'] = name_segments(df['cluster'], df)
g = df.groupby('segment')
profile = g[num].median().round(2)
profile.insert(0, 'listings', g.size())
profile['limited %'] = (g['available_limited'].mean() * 100).round(0)
profile['value share %'] = (g['est_sales_value'].sum() / df['est_sales_value'].sum() * 100).round(1)
print('Medians unless stated; value share = share of estimated sales value (price x units sold)\n')
print(profile.reindex(SEGMENTS).to_string())


# In[27]:


plt.figure(figsize=(7, 4.5))
for seg in SEGMENTS:
    m = (df['segment'] == seg).values
    plt.scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.6, label=seg)
plt.xlabel('PC1 (sales and stock)'); plt.ylabel('PC2 (days since update)')
plt.title('The five segments on the first two principal components'); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()


# ### 4.3 Four or five clusters? (Figures 27 to 29)

# In[28]:


k4 = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit_predict(X)
print(pd.DataFrame({'k = 4': scores(X, k4), 'k = 5': scores(X, df['cluster'])}).round(3))
print('\nWhere the five segments go when k = 4:')
moves = pd.crosstab(df['segment'], k4,
                    rownames=['five-cluster segment'], colnames=['four-cluster group'])
print(moves.reindex(SEGMENTS))


# In[29]:


total_value = df['est_sales_value'].sum()

def describe(groups):
    g = df.groupby(groups)
    return pd.DataFrame({'listings': g.size(),
                         'price Q1': g['price'].quantile(.25),
                         'median price': g['price'].median(),
                         'price Q3': g['price'].quantile(.75),
                         'median sold': g['sold'].median(),
                         'value share %': g['est_sales_value'].sum() / total_value * 100}).round(2)
print('Five clusters:'); print(describe(df['segment']).reindex(SEGMENTS).to_string())
print('\nFour clusters (groups 0 to 3):'); print(describe(k4).to_string())


# In[30]:


# Is five clusters consistently better than four? Refit both on 200 bootstrap samples
# (listings drawn at random with replacement) and compare the three indices each time
rng = np.random.default_rng(SEED)
diffs = []
for _ in range(200):
    Xb = X.iloc[rng.integers(0, len(X), len(X))].reset_index(drop=True)
    s4 = scores(Xb, KMeans(n_clusters=4, n_init=20, random_state=SEED).fit_predict(Xb))
    s5 = scores(Xb, KMeans(n_clusters=5, n_init=20, random_state=SEED).fit_predict(Xb))
    diffs.append({m: s5[m] - s4[m] for m in s4})
diffs = pd.DataFrame(diffs)
five_better = pd.Series({'silhouette': (diffs['silhouette'] > 0).mean(),
                         'davies_bouldin': (diffs['davies_bouldin'] < 0).mean(),   # lower is better
                         'calinski_harabasz': (diffs['calinski_harabasz'] > 0).mean()})
print(pd.DataFrame({'mean difference (5 - 4)': diffs.mean(), '2.5%': diffs.quantile(.025),
                    '97.5%': diffs.quantile(.975), 'samples where 5 is better': five_better}).round(3))


# ### 4.4 Sensitivity to stock assumptions (Figure 30)

# In[31]:


censored = df['available_10plus'] == 1      # stored as 10, but only known to be 'more than 10'
# scenario: (value for the 297 censored listings, value for the 64 imputed listings)
scenarios = {'Baseline': (10, 9), 'S1: censored = 15': (15, 9), 'S2: missing = 4': (10, 4),
             'S3: both (15 and 4)': (15, 4), 'S4: censored = 50': (50, 9)}

rows = []
for name, (censored_value, missing_value) in scenarios.items():
    d = df.copy()
    d.loc[censored, 'available'] = censored_value
    d.loc[stock_imputed, 'available'] = missing_value
    scaled_s = pd.DataFrame(StandardScaler().fit_transform(np.log1p(d[num])), columns=num)
    Xs = pd.concat([scaled_s, cat_dummies], axis=1)      # same steps as Section 3.3
    lab = KMeans(n_clusters=5, n_init=20, random_state=SEED).fit_predict(Xs)
    seg = name_segments(lab, d)                          # same naming rule as Section 4.2
    rows.append({'scenario': name, 'same segment %': round((seg == df['segment']).mean() * 100, 1),
                 'ARI': round(adjusted_rand_score(df['cluster'], lab), 3),
                 'silhouette': round(silhouette_score(Xs, lab), 3),
                 **seg.value_counts().reindex(SEGMENTS, fill_value=0).to_dict()})
print(pd.DataFrame(rows).set_index('scenario').to_string())

print('\nWhere listings move under S4 (rows = baseline segment):')
moves = pd.crosstab(df['segment'], seg, colnames=['segment under S4'])
print(moves.reindex(index=SEGMENTS, columns=SEGMENTS, fill_value=0))

moved_in = (seg == 'Bulk stockists') & (df['segment'] != 'Bulk stockists')
print(f'\nListings moving into Bulk stockists under S4: {moved_in.sum()}',
      f'| share that are censored: {censored[moved_in].mean():.0%}')


# # Task 2: Figures 31 and 32

# In[32]:


prof = profile.sort_values('value share %', ascending=False)
listing_share = prof['listings'] / prof['listings'].sum() * 100
fig, ax = plt.subplots(figsize=(8.5, 4.6))
y, h = np.arange(len(prof)), 0.38
ax.barh(y + h / 2, listing_share, h, label='Share of listings', color='#9DB4D4')
ax.barh(y - h / 2, prof['value share %'], h, label='Share of estimated sales value', color='#1F3864')
for yy, a, b in zip(y, listing_share, prof['value share %']):
    ax.text(a + 0.8, yy + h / 2, f'{a:.0f}%', va='center', fontsize=8)
    ax.text(b + 0.8, yy - h / 2, f'{b:.0f}%', va='center', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(prof.index); ax.invert_yaxis()
ax.set_xlabel('Percentage (%)'); ax.set_title('Where listings sit and where estimated sales value comes from')
ax.legend(loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_task2_value.png', dpi=150); plt.show()


# In[33]:


offsets = {'Bestsellers': (42, 0, 'left'), 'Bulk stockists': (-24, 0, 'right'), 'Stale listings': (36, 8, 'left'),
           'Budget slow movers': (0, -40, 'center'), 'Premium slow movers': (0, 40, 'center')}
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(profile['price'], profile['sold'], s=profile['listings'] * 6, color='#1F3864', alpha=0.75, edgecolor='white')
for seg, r in profile.iterrows():
    dx, dy, ha = offsets[seg]
    ax.annotate(f"{seg}\n{int(r['listings'])} listings, {r['value share %']:.0f}% of sales value", (r['price'], r['sold']),
                textcoords='offset points', xytext=(dx, dy), ha=ha, va='center', fontsize=8.5)
ax.set_yscale('log'); ax.set_ylim(3, 3000); ax.set_xlim(0, 95)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.set_xlabel('Median price ($)'); ax.set_ylabel('Median units sold (log scale)')
ax.set_title('The five segments by typical price and sales'); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_task2_segments.png', dpi=150); plt.show()

