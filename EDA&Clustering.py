

# # SUBMISSION

# In[1]:
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import warnings
import os
# In[2]:
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import to_date, unix_timestamp, year, month, hour, to_timestamp, udf
from pyspark.sql.functions import concat, col, lit, udf, split
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
# In[3]:
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
# In[4]:
from pyspark.sql import SparkSession
spark = SparkSession     .builder     .appName("fire analysis")     .config("spark.some.config.option", "some-value")     .getOrCreate()
# In[5]:
df_fire = spark.read.format("csv").option("header", "true").load("C:\\Users\\manoja\\Downloads\\FireDepartmentCalls1.csv")
df_fire.createOrReplaceTempView("sf_fire")
# In[6]:
df_fire.printSchema()
# In[7]:
df_fire.count()
# In[8]:
df_fire.head(3)
# In[9]:
df_fire.describe().toPandas().transpose()
# In[10]:
df = (df_fire.withColumn("Received DtTm", unix_timestamp("Received DtTm", "MM/dd/yyyy hh:mm:ss a").cast('timestamp'))
        .withColumn("Year", year(col("Received DtTm")))
        .withColumn("Month", month(col("Received DtTm"))).withColumn("hour", hour(col("Received DtTm"))))

# In[11]:
df=df.withColumn('X',df['X'].cast("float").alias('X'))
df=df.withColumn('Y',df['Y'].cast("float").alias('Y'))
df=df.withColumn('Call Number',df['Call Number'].cast("int").alias('Call Number'))
# In[12]:
df.printSchema()
# In[13]:
df = df.filter(df['Call Type'] != 'Medical Incident')
q1_result = df.groupBy('Call Type').count().orderBy('count', ascending=False)
q1_result.show()
# In[14]:
# Show the number of incidents for different fire emergency call type
q1 = q1_result.toPandas()
print(q1)
# In[15]:
# Visualize top ten fire department incidents in San Francisco
import time
start_time = time.time()
fig, ax = plt.subplots(figsize=(18, 10))
sns.barplot(x='Call Type', y='count', data=q1.loc[:9, :], ax=ax)
ax.set_title('Top ten fire incident types', fontsize=15)
ax.set_xlabel('Incident', fontsize=10)
ax.set_ylabel('Count', fontsize=15)
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[16]:
start_time = time.time()
q2_result = df.groupBy('Neighborhooods - Analysis Boundaries').count().orderBy('count', ascending=False)
q2_result.show()
print("--- %s seconds ---" % (time.time() - start_time))
# In[17]:
# The number of fire incidents for different neighborhoood
q2 = q2_result.toPandas()
print(q2)
# In[18]:
start_time = time.time()
# Visualize fire incidents for different neighbourhood
fig, ax = plt.subplots(figsize=(18, 10))
sns.barplot(x='Neighborhooods - Analysis Boundaries', y='count', data=q2.loc[:9, :], ax=ax)
ax.set_title('Fire incidents for different districts', fontsize=15)
ax.set_xlabel('Incident', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[19]:
start_time = time.time()
# Visualize incidents of different categories for different district, here we take the top 15 fire emergnecy categories.
fig, ax = plt.subplots(figsize=(18, 10))
df_Pd = df.groupBy('Call Type', 'Neighborhooods - Analysis Boundaries').count()
q2_2 = df_Pd.toPandas()
table = pd.pivot_table(q2_2, values='count', index=['Neighborhooods - Analysis Boundaries'], columns=['Call Type'])
table = table[q1['Call Type'][:15].values]
order = q2['Neighborhooods - Analysis Boundaries'].values
table.loc[order].plot.bar(stacked=True, ax=ax, edgecolor = "none", cmap='tab10').legend(bbox_to_anchor=(0.95, 0.9))
ax.set_title('Fire incidents for different districts', fontsize=15)
ax.set_xlabel('Incident', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[20]:
start_time = time.time()
fig, ax = plt.subplots(figsize=(12, 5))
for year in ['2000', '2005', '2010', '2015','2020', '2022']:
    df_month = df.filter(df['Year'] == year).groupBy('Month').count().orderBy('Month', ascending=True)
    df_month = df_month.toPandas()
    ax.plot(df_month['Month'], df_month['count'],label = year)
    ax.set_title('Fire incidents from 2015 to 2022')
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    ax.legend()
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[21]:
start_time = time.time()
fig, ax = plt.subplots(figsize=(12, 5))
for date in ['12/15/2015','12/15/2016','12/15/2017','12/15/2018', '12/15/2019', '12/15/2020','12/15/2021','12/15/2022']:
    df_hour = df.filter(df['Call Date'] == date).groupBy('Hour').count().orderBy('Hour', ascending=True)
    df_hour = df_hour.toPandas()
    ax.plot(df_hour['Hour'], df_hour['count'], label=date)
    ax.set_title('Fire emergency incidents from 2015 to 2022 (winter)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[22]:
start_time = time.time()
fig, ax = plt.subplots(figsize=(12, 5))
for date in ['12/15/2015','12/15/2016','12/15/2017','12/15/2018', '12/15/2019', '12/15/2020']:
    df_hour = df.filter(df['Call Date'] == date).groupBy('Hour').count().orderBy('Hour', ascending=True)
    df_hour = df_hour.toPandas()
    ax.plot(df_hour['Hour'], df_hour['count'], label=date)
    ax.set_title('Fire emergency incidents from 2015 to 2020 (winter)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[23]:
start_time = time.time()
fig, ax = plt.subplots(figsize=(12, 5))
for date in ['07/01/2015', '07/01/2016', '07/01/2017','07/01/2018','07/01/2019','07/01/2020','07/01/2021','07/01/2022']:
    df_hour = df.filter(df['Call Date'] == date).groupBy('Hour').count().orderBy('Hour', ascending=True)
    df_hour = df_hour.toPandas()
    ax.plot(df_hour['Hour'], df_hour['count'], label=date)
    ax.set_title('Fire emergency incidents from 2015 to 2022 (Summer)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[24]:
start_time = time.time()
fig, ax = plt.subplots(figsize=(12, 5))
for date in ['07/01/2015','07/01/2016','07/01/2017','07/01/2018','07/01/2019','07/01/2020']:
    df_hour = df.filter(df['Call Date'] == date).groupBy('Hour').count().orderBy('Hour', ascending=True)
    df_hour = df_hour.toPandas()
    ax.plot(df_hour['Hour'], df_hour['count'], label=date)
    ax.set_title('Fire emergency incidents from 2015 to 2020 (Summer)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[25]:
start_time = time.time()
# Visualize fire related event w.r.t category and time (hour) top-3 district
fig, ax = plt.subplots(3, 1, figsize=(15, 20))
area = ['Tenderloin', 'Financial District/South Beach', 'Mission']
for i in range(3):
    df_area = df.filter(df['Neighborhooods - Analysis Boundaries'] == area[i]).groupBy('Call Type', 'Hour').count()
    q6 = df_area.toPandas()
    table = pd.pivot_table(q6, values='count', index=['Hour'], columns=['Call Type'])
    table = table[q1['Call Type'][:7].values]
    table.plot.bar(stacked=True, ax=ax[i], edgecolor = "none", cmap='Paired').legend(bbox_to_anchor=(0.35, 1))
    ax[i].set_title(area[i])
    ax[i].set_xlabel('Hour')
    ax[i].set_ylabel('Count')
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[26]:
start_time = time.time()
df_q7 = df.filter(df['Year'] != 2019).groupBy('Year', 'Call Type').count().orderBy('Call Type', 'Year', ascending=True)
df_q7 = df_q7.toPandas()
sum_year = df_q7.groupby('Year').sum()
df_q7 = df_q7.merge(sum_year, how='left', left_on='Year', right_index=True)
df_q7['percentage'] = df_q7['count_x'] / df_q7['count_y']
df_q7.head(10)
print("--- %s seconds ---" % (time.time() - start_time))
# In[27]:
df_q7.head(10)
# In[28]:
start_time = time.time()
# Visualize the percentage of different fire related incidents over time 
fig, ax = plt.subplots(5, 1, figsize=(15, 20))
j = 0
for category in (q1['Call Type'][i] for i in [0, 1, 5, 6, 8]):
    df_temp = df_q7[df_q7['Call Type'] == category]
    ax[j].plot(df_temp['Year'], df_temp['percentage'],label = category)
    ax[j].set_xlabel('Year')
    ax[j].set_ylabel('Percentage')
    ax[j].legend()
    j += 1
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[29]:
start_time = time.time()
q8_result = df.groupBy('X', 'Y').count()
q8 = q8_result.toPandas()
print("--- %s seconds ---" % (time.time() - start_time))
# In[30]:
start_time = time.time()
# Visualize the spatial distribution of fire related incidents
fig, ax = plt.subplots(figsize=(15, 14))
ax.scatter(q8['X'], q8['Y'], s=q8['count']/100, alpha=0.5, edgecolors='grey')
ax.set_title('Spatial distribution of fire incidents')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_ylim([37.67, 37.85])
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))
# In[31]:
start_time = time.time()
# Perfrom clustering analysis based on the number of top 5 fire incident categories at different location.
df_cluster = df.groupBy('X', 'Y').pivot('Call Type').count()
df_cluster = df_cluster.na.fill(0)
print("--- %s seconds ---" % (time.time() - start_time))
# In[32]:
FEATURES_COL = ['Structure Fire', 'Alarms', 'Outside Fire', 'Vehicle Fire', 'Gas Leak (Natural and LP Gases)']
# In[33]:
vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(df_cluster).select('X', 'Y', 'features')
df_kmeans.show(20)
# In[34]:
# Trains a k-means model
kmeans = KMeans().setK(5).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()
# In[35]:
transformed = model.transform(df_kmeans).select('X', 'Y', 'prediction')
rows = transformed.collect()
print(rows[:5])
# In[36 ]:
# Clustering result
start_time = time.time()
df_pred = sqlContext.createDataFrame(rows)
# Merge clustering results with fire events count
df_pred = df_pred.toPandas()
df_pred = df_pred.merge(q8, on=['X','Y'])
# Visulize clustering results (5 clusters)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df_pred['X'], df_pred['Y'], s=df_pred['count']/100, c=df_pred['prediction'], cmap='rainbow', alpha=0.5, edgecolors='grey')
ax.set_title('Clustering of fire incidents')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_ylim([37.67, 37.85])
display(fig)
print("--- %s seconds ---" % (time.time() - start_time))

