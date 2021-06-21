#!/usr/bin/env python
# coding: utf-8

# ![pics.png](attachment:pics.png)

# ### MSBA 326 Final Project: Customer Segmentation using K Means Clustering 

# In[1]:


# Importing libraries                  
from __future__ import division
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 
from pandas import read_csv
import seaborn as sb
from textwrap import wrap
import statsmodels.api as sm

from datetime import datetime, timedelta,date
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


import plotly as py
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.graph_objs as go
import plotly.offline as py


# In[4]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[5]:


# Loading the data
filename = 'customer_segmentation.csv'
df = read_csv(filename, encoding="ISO-8859-1")


# ### Data Exploration and Preparation

# In[6]:


df.head(3)


# In[7]:


df.isna().sum()


# In[8]:


df = df.dropna()


# In[9]:


df.isna().sum()


# In[10]:


df.dtypes


# In[11]:


#converting the type of Invoice Date Field from string to datetime.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#creating YearMonth field for the ease of reporting and visualization
df['InvoiceYearMonth'] = df['InvoiceDate'].map(lambda date: 100*date.year + date.month)


# In[12]:


df.dtypes


# In[13]:


# Let's see which country gets maximum number of orders
fig = px.histogram(df, x="Country")
fig.update_layout(xaxis = go.layout.XAxis(tickangle = 45))
fig.show()


# As per above graph, United Kingdom made up the majority of the sales figure. This is not surprizing given that this is a store is based in the UK. 
# 
# Since majority of data is from the UK, We will work on segmenting market and all our analyses using UK data only. 
# 
# Creating a new dataframe with UK records 

# In[14]:


ukdf=df.loc[df['Country'] == 'United Kingdom']


# #### Finding Outliers

# In[15]:


# Creating a new column for Total amount per transaction
ukdf['Revenue'] = ukdf['Quantity']*ukdf['UnitPrice']


# In[16]:


# Taking a look at negative values.
NegTrans = ukdf[ukdf['Revenue'] < 0]
NegTrans.head(5)


# A look at the quantity, unitPrice, Revenue values also reveal some negative values. These can be intepreted as refunds. As the total quantity and total revenue values of refunds and the original purchase cancel together, we will leave them as it is.

# In[17]:


# Dropping unnecessary column
ukdf = ukdf.drop(['StockCode'], axis =1)


# In[18]:


ukdf.shape


# In[19]:


# Descriptive Analysis before removing outliers
ukdf[['Quantity', 'UnitPrice','Revenue']].describe()


# #### 1. Unit Price

# In[20]:


ukdf.groupby('Description').mean()['UnitPrice'].nlargest()


# A look at the highest mean values bring up some suspect "products". We will dig deeper into above products to detect outliers.

# In[21]:


df[df['Description']== 'DOTCOM POSTAGE']['UnitPrice'].describe()


# In[22]:


df[df['Description']== 'Manual']['UnitPrice'].describe()


# In[23]:


df[df['Description']== 'CRUK Commission'].head()


# In[24]:


df[df['Description']== 'Discount'].head()


# We found below 4 items that have unusual unit price and also does not seem like a product:
# 
# 1. DOTCOM POSTAGE - seems to indicate the amount spent by the customer on postage
# 
# 2. CRUK Commission - fee paid out to an external cancer research organization
# 
# 3. Manual - refers to manual services rendered with the purchase like furniture assembly
# 
# 4. Discount 
# 
# All these items are not direct indicator of sales and are heavily skewing the data. So, we will remove them from our dataset.

# In[25]:


removeitems = ['DOTCOM POSTAGE', 'CRUK Commission', 'Manual','Discount']
ukdf = ukdf[~ukdf['Description'].isin(removeitems)]


# In[26]:


# Let's check if we still have any rows with unusual unitprice i.e. greater than 2K
ukdf[ukdf.UnitPrice > 2000]


# In[27]:


removeinvoice1 = ['C551685', '551697']
ukdf = ukdf[~ukdf['InvoiceNo'].isin(removeinvoice1)]


# In[28]:


# So, we deleted 1315 rows, but since we still have 494K of data it should be enough.
ukdf.shape


# In[29]:


ukdf.head()


# #### Let's plot frequency distribution of some of the relevant columns
# 
# #### 1. UnitPrice

# In[30]:


fig = px.histogram(ukdf, x="UnitPrice",
                 labels={
                     "UnitPrice": "UnitPrice",
                     },
                title="UnitPrice Frequency")
fig.show()


# In[31]:


# As we see most of the unit prices are under 50, we will take a closer look
ukdf1=ukdf.query('UnitPrice < 50')['UnitPrice']
fig = px.histogram(ukdf1, x="UnitPrice",
                 labels={
                     "UnitPrice": "UnitPrice",
                     },
                title="UnitPrice Frequency")
fig.show()


# #### 2. Quantity

# By running a quick filter, we see that there are only 106 rows which have quantities above 1000 and below -1000. Upon closer look we also found they mostly lie in small ticket items and most of them were returned, indicating they were bought by mistake. Thus we will remove these outliers.

# In[32]:


ukdf[(ukdf['Quantity'] > 1000) | (ukdf['Quantity'] < -1000)]


# In[33]:


# We will remove all the quantities above 1K and their corresponding return transaction of less than -1K.
ukdf = ukdf[(ukdf['Quantity'] > -1000)]


# In[34]:


ukdf = ukdf[(ukdf['Quantity'] < 1000)]


# In[35]:


ukdf2 = ukdf[ukdf.Quantity > 0]        # leaving out negative values as they are return items
fig = px.histogram(ukdf2, x="Quantity",
                 labels={
                     "Quantity": "Quantity",
                     },
                title="Quantity Frequency")
fig.show()


# In[36]:


# Most number of quantities lie below 100, so taking a closer look at the histogram
ukdf3=ukdf2.query('Quantity < 100')['Quantity']
fig = px.histogram(ukdf3, x="Quantity",
                 labels={
                     "Quantity": "Quantity",
                     },
                title="Quantity Frequency")
fig.show()


# #### Finding top ten selling items by their total sales

# In[37]:


sales_order = ukdf.groupby('Description').sum()['Revenue'].nlargest(10)

plt.figure(figsize = (30,10))
ax = sb.barplot(x = sales_order.index, y = sales_order.values, palette = 'viridis')
ax.set_xlabel('Product Description', fontsize = 20)
ax.set_ylabel('Total Sales', fontsize = 20)
ax.set_title('Top 10 Selling Products', fontsize = 30)

labels = [ '\n'.join(wrap(l, 15)) for l in sales_order.index ]
ax.set_xticklabels(labels, fontsize = 15)

value_labels = []
for x in sales_order.values:
    value_labels.append(str(int(x/1000))+' k')

for p, label in zip(ax.patches, value_labels):
    ax.annotate(label, (p.get_x() + 0.26, p.get_height() + 2), fontsize = 15)


# It seems 3 tier cakestand is the highest revenue generating item.

# In[38]:


ukdf.shape   # Even after cleaning up the outliers, we still have 490K records so we should be good to go.


# In[39]:


# Descriptive Analysis of relevant columns after removing outliers
ukdf[['Quantity', 'UnitPrice','Revenue']].describe()


# ### Customer segmentation using RFM Clustering

# In[40]:


#creating a generic user dataframe to keep CustomerID and new segmentation scores
ukdf_user = pd.DataFrame(ukdf['CustomerID'].unique())
ukdf_user.columns = ['CustomerID']
ukdf_user.head()


# #### Recency Analysis

# In[41]:


# Creating a dataframe with max purchase date for each customer 
uk_max_purchase = ukdf.groupby('CustomerID').InvoiceDate.max().reset_index()
uk_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
uk_max_purchase.head(3)


# In[42]:


# Comparing the last transaction of the dataset with last transaction dates of the individual customer IDs.
uk_max_purchase['Recency'] = (uk_max_purchase['MaxPurchaseDate'].max() - uk_max_purchase['MaxPurchaseDate']).dt.days
uk_max_purchase.head()


# In[43]:


#merging this dataframe to the new user dataframe
ukdf_user = pd.merge(ukdf_user, uk_max_purchase[['CustomerID','Recency']], on='CustomerID')
ukdf_user.head()


# In[44]:


# Plotting Recency histogram
fig = px.histogram(ukdf_user, x="Recency",
                 labels={
                     "Recency": "Last Purchase days",
                     },
                title="Recency")
fig.show()


# We will now apply K-means clustering to assign a recency score. We start with specifying the number of clusters we need for K-means algorithm. To find it out, we will apply Elbow Method that tells the optimal cluster number for optimal inertia. Code snippet and Inertia graph are as follows:

# In[45]:


from sklearn.cluster import KMeans

sse={} # error
uk_recency = ukdf_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(uk_recency)
    uk_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[46]:


#  3 looks like an optimal number. We will try expand our scope and use 4 for our further analysis. 


# In[47]:


#building 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
ukdf_user['RecencyCluster'] = kmeans.fit_predict(ukdf_user[['Recency']])


# In[48]:


ukdf_user.head()


# In[49]:


ukdf_user.groupby('RecencyCluster')['Recency'].describe()


# Right now each customer is randomly assigned to each cluster. However, clusters are not ordered because cluster 2 customers are active than cluster 0 but older than both cluster 1 & 3. Let us order the clusters according to most recent transactions by finding the mean of recency value corresponding to each cluster.

# In[50]:


# Building the function 
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

ukdf_user = order_cluster('RecencyCluster', 'Recency',ukdf_user,False)


# In[51]:


ukdf_user.head()


# In[52]:


ukdf_user.groupby('RecencyCluster')['Recency'].describe()


# #### Frequency Analysis 

# In[53]:


# To create frequency clusters, we need to find total number of orders for each customer. 
#get order counts for each user and create a dataframe with it
uk_frequency = ukdf.groupby('CustomerID').InvoiceDate.count().reset_index()
uk_frequency.columns = ['CustomerID','Frequency']


# In[54]:


uk_frequency.head() # number of orders per customer


# In[55]:


# We will add this data to our generic dataframe

ukdf_user = pd.merge(ukdf_user, uk_frequency, on='CustomerID')

ukdf_user.head()


# In[56]:


# Plotting Recency histogram
fig = px.histogram(ukdf_user, x="Frequency",
                 labels={
                     "Frequency": "Frequency of purchase",
                     },
                title="Frequency")
fig.show()


# In[57]:


# The maximum frequencies are below 1000. Visualizing frequencies below 1000.
df2=ukdf_user.query('Frequency < 1000')['Frequency']
fig = px.histogram(df2, x="Frequency",
                 labels={
                     "Frequency": "Frequency of purchase",
                     },
                title="Frequency")
fig.show()


# In[58]:


# Frequency clusters

# Determining the right number of clusters for K-Means by elbow method
from sklearn.cluster import KMeans

sse={} # error
uk_recency = ukdf_user[['Frequency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(uk_recency)
    uk_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[59]:


# Applying k-Means

kmeans=KMeans(n_clusters=4)
ukdf_user['FrequencyCluster']=kmeans.fit_predict(ukdf_user[['Frequency']])

#ordering the frequency cluster
ukdf_user = order_cluster('FrequencyCluster', 'Frequency', ukdf_user, True )
ukdf_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[60]:


# Max frequency can be seen in cluster 3, least frequency cluster is cluster 0.


# #### Monetary Value

# In[61]:


# We will now cluster our customers based on revenue. 
#calculate revenue for each customer
uk_revenue = ukdf.groupby('CustomerID').Revenue.sum().reset_index()


# In[62]:


uk_revenue.head()


# In[63]:


# merging uk_revenue with our main dataframe
ukdf_user = pd.merge(ukdf_user, uk_revenue, on='CustomerID')


# In[64]:


ukdf_user.head(3)


# In[65]:


# Visualizing with the histogram on a reduced scale 
#plot the histogram
df4=ukdf_user.query('Revenue < 10000')['Revenue']
fig = px.histogram(df4, x="Revenue",
                 labels={
                     "Revenue": "Monetary value of purchase",
                     },
                title="Revenue")
fig.show()


# In[66]:


# Using elbow method to find out the optimum number of clusters for K-Means

from sklearn.cluster import KMeans

sse={} # error
uk_recency = ukdf_user[['Revenue']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(uk_recency)
    uk_recency["clusters"] = kmeans.labels_  # cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ # sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[67]:


# Elbow method suggests optimal clusters can be 3 or 4. We will take 4 as the number of clusters

#apply clustering
kmeans = KMeans(n_clusters=4)
ukdf_user['RevenueCluster'] = kmeans.fit_predict(ukdf_user[['Revenue']])

#order the cluster numbers
ukdf_user = order_cluster('RevenueCluster', 'Revenue',ukdf_user,True)

#show details of the dataframe
ukdf_user.groupby('RevenueCluster')['Revenue'].describe()


# In[68]:


# It seems that cluster 3 has max revenue and cluster 0 has lowest revenue


# In[69]:


# Overall scores

# Now we have scores (cluster numbers) for recency, frequency & revenue. We will now create an overall score.
# Calculating overall score and use mean() to see details
ukdf_user['OverallScore'] = ukdf_user['RecencyCluster'] + ukdf_user['FrequencyCluster'] + ukdf_user['RevenueCluster']
ukdf_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[70]:


# It seems that customer with score 7 is our best customer and one with score 0 is least attractive customer.


# In[71]:


# Analysing low value and high value customers 
ukdf_user['Segment'] = 'Low-Value'
ukdf_user.loc[ukdf_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
ukdf_user.loc[ukdf_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[72]:


ukdf_user.head()


# #### Visualizing segments with scatter plots

# In[73]:


#Revenue vs Frequency
uk_graph = ukdf_user.query("Revenue < 10000 and Frequency < 1000")

plot_data = [
    go.Scatter(
        x=uk_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=uk_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=uk_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'High-Value'")['Frequency'],
        y=uk_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[74]:


#Revenue vs Recency

uk_graph = ukdf_user.query("Revenue < 10000 and Frequency < 1000")

plot_data = [
    go.Scatter(
        x=uk_graph.query("Segment == 'Low-Value'")['Recency'],
        y=uk_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=uk_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'High-Value'")['Recency'],
        y=uk_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer="notebook")


# In[75]:


# Recency vs Frequency Visualization 

uk_graph = ukdf_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=uk_graph.query("Segment == 'Low-Value'")['Recency'],
        y=uk_graph.query("Segment == 'Low-Value'")['Frequency'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=uk_graph.query("Segment == 'Mid-Value'")['Frequency'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=uk_graph.query("Segment == 'High-Value'")['Recency'],
        y=uk_graph.query("Segment == 'High-Value'")['Frequency'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer="notebook")


# ### We used K means clustering algorithm and created customer clusters that we will use for segmenting our market in the United Kingdom.
# 
# The clusters are created using Recency, Frequency, and Monetary Value (revenue) framework, popular in marketing.
# From the analysis and visualization above, we now know which customers we need to retain (high value) and which segment does not require our attention (low value).
# 
# We acknowledge the work of Shailaja Gupta that we used for our study. Reference:
# Gupta, S. (n.d.). Customer Segmentation: RFM Clustering. Retrieved from https://kaggle.com/shailaja4247/customer-segmentation-rfm-clustering
# 
# The picture on the top is taken from an online article. Reference:
# Ong, Adeline(2020, Mar 16). Segmenting Customers using K-Means, RFM and Transaction Records. Retrieved from https://towardsdatascience.com/segmenting-customers-using-k-means-and-transaction-records-76f4055d856a 

#                                      *** End of the Project***
