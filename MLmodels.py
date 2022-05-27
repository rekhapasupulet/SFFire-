
# In[1]:
import pyspark.sql.functions as F
from pyspark.sql.window import Window
newdf= newdf.withColumn("Call_Type_num", F.dense_rank().over(Window.orderBy("Call Type Group")))
# In[2]:
newdf.select('Call Type Group').distinct().show(35, False)
# In[3]:
newdf.select('Call_Type_num').distinct().show(35, False)
# In[4]:
import pyspark.sql.functions as F
newdf = newdf.filter(
               (F.col("Call Type Group") != "null") 
               )
# In[5]:
newdf.select('Call Type Group').distinct().show(35, False)
# In[6]:
newdf.select('Call_Type_num').distinct().show(35, False)
# In[7]:
min = newdf.agg({"X": "min"}).collect()[0][0]
max = newdf.agg({"X": "max"}).collect()[0][0]
df_scaled = newdf.withColumn('X_scaled', (col('X') - min)/max)
# In[8]:
min = newdf.agg({"Y": "min"}).collect()[0][0]
max = newdf.agg({"Y": "max"}).collect()[0][0]
df_scaled2 = df_scaled.withColumn('Y_scaled', (col('Y') - min)/max)
# In[9]:
min = newdf.agg({"hour": "min"}).collect()[0][0]
max = newdf.agg({"hour": "max"}).collect()[0][0]
df_scaled3 = df_scaled2.withColumn('hour_scaled', (col('hour') - min)/max)
# In[10]:
min = newdf.agg({"X": "min"}).collect()[0][0]
max = newdf.agg({"X": "max"}).collect()[0][0]
df_scaled = newdf.withColumn('scaled_results', (col('X') - min)/max)
# In[11]:
df_scaled3.printSchema()
# In[12]:
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = [ 'X_scaled','Y_scaled','hour_scaled','Month'], outputCol = 'features')
df_scaled= df_scaled3.fillna(0)
vfire_df = vectorAssembler.transform(df_scaled3)
vfire_df = vhouse_df.select(['features', 'Call_Type_num'])
vfire_df.show()
# In[13]:
splits = vfire_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
# In[14]:
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'Call_Type_num')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
dt_evaluator = MulticlassClassificationEvaluator(
    labelCol="Call_Type_num", predictionCol="prediction", metricName="accuracy")
accuracy = dt_evaluator.evaluate(dt_predictions)
# In[15]:
print(" Decision Tree Accuracy = %g" % accuracy)
# In[16]:
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol ='features', labelCol = 'Call_Type_num')
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
rf_evaluator = MulticlassClassificationEvaluator(
    labelCol="Call_Type_num", predictionCol="prediction", metricName="accuracy")
accuracy = rf_evaluator.evaluate(rf_predictions)
# In[17]:
print("Random Forest Classifier Accuracy = %g" % accuracy)
# In[18]:
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(featuresCol ='features', labelCol = 'Call_Type_num')
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
gbt_evaluator = MulticlassClassificationEvaluator(
    labelCol="Call_Type_num", predictionCol="prediction", metricName="accuracy")
accuracy = gbt_evaluator.evaluate(gbt_predictions)
# In[19]:
print("GBTClassifier Accuracy = %g" % accuracy)
