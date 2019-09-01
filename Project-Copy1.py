
# coding: utf-8

# # Import Modules

# #use to get into rcc
# lab 3
# export h='hostname -i'
# echo sh
# Pyspark_Driver_ptyhon = jupyter pyspark_driver_python_opts="notebook --no-browser --ip=$h" pyspark
# 
# #use for batch files below
# module load spark/2.3.0
# 
# # To use multiple notdes
# start-spark-slurm.sh
# export master=spark://$hostname:7077
# spark-submit --master perceptron.py
# 
# Bigdl lab 8
# module load Anaconda3/2019.03
# cd /software/Anaconda3-5.1.0-hadoop/bin/pip
# 
# /software/Anaconda4-5.1.0-hadoop/bin/pip install 

# In[2]:


# Regular modules
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyspark
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re 
import string
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import folium


#spark sql imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_timestamp, col, round, month, year, udf, date_format, to_date, datediff, lower


#spark ML imports
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, Word2Vec, OneHotEncoder, StringIndexer, OneHotEncoderEstimator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import bigdl
from pyspark.mllib.classification import LogisticRegressionWithLBFGS,SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
#!conda install systemml keras tensorflow 
#from systemml.mllearn import Keras2DML
#!conda update -n base -c defaults conda
#!conda install -c johnsnowlabs spark-nlp
#!y
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.embeddings import *


#change configuration settings on Spark 
spark = SparkSession.builder.master('yarn-client').appName("local[*]").getOrCreate()
conf = spark.sparkContext._conf.setAll([("spark.sql.crossJoin.enabled", "true"),('spark.executor.memory', '8g'), ('spark.app.name', 'Spark Updated Conf'), ('spark.executor.cores', '32'), ('spark.cores.max', '32'), ('spark.driver.memory','15g'),("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.1.0")])
sqlContext = pyspark.SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)


# In[3]:


df_income = spark.read.csv('data/project/kaggle_income.csv',header=True, inferSchema="true")


# In[4]:


df_income.printSchema()


# ### Where are the least wealthy and wealthiest places to live?

# In[5]:


df_income2 = df_income.groupby(["State_Name","City"]).agg({"Median":"avg"})
df_income2 = df_income2.withColumn("avg(Median)", round(df_income2["avg(Median)"],2).alias("median"))
df_income2.sort(col("avg(Median)").desc()).show(5)
df_income2.sort(col("avg(Median)").asc()).show(5)


# In[6]:


df_income3 = df_income.groupby(["City","Zip_Code"]).agg({"Median":"avg"}).alias("median1")
df_income3 = df_income2.select(col("City").alias("City_match"), col("avg(Median)").alias("Median_match"))
df_income4 = df_income2.join(df_income3, df_income2.City== df_income3.City_match, how="left")
df_income5 = df_income4.groupby(["State_Name","City"]).agg({"avg(Median)":"Mean"})
df_income5.show(20)


# In[7]:


df_income2.show(200)


# In[8]:


df_income1 = df_income.toPandas()
df_income1['latlon'] = df_income1.apply(lambda x: '('+str(x.Lat)+','+str(x.Lon)+')',axis=1 )


# In[9]:


df_income1.columns


# # Import all yelp data

# Find out which states are people yelping the most and make a heat map of the USA.

# In[10]:


df_bus = sqlContext.read.json('user/adhamsuliman/data/business.json').dropna(thresh=1,subset=('state','city','business_id',"longitude","latitude"))
df_bus.printSchema()


# In[11]:


df_bus.groupby("state").count().sort(col("count").desc()).show(50)


# In[12]:


df_checkin = sqlContext.read.json('user/adhamsuliman/data/checkin.json').dropna(thresh=1, subset='business_id')
df_checkin.printSchema()
df_review = sqlContext.read.json('user/adhamsuliman/data/review.json').dropna(thresh=1, subset=('stars','business_id'))
df_review.printSchema()
df_tip = sqlContext.read.json('user/adhamsuliman/data/tip.json').dropna(thresh=1, subset=('business_id','user_id'))
df_tip.printSchema()
#problems with user
#df_user = sqlContext.read.json('user/adhamsuliman/data/user.json')
#df_user.printSchema()


# In[13]:


df_restaurants = df_bus.filter(df_bus.categories.like('%Restaurants%')|df_bus.categories.like('%Food%'))
df_restaurants_match = df_restaurants.groupby(["City"]).agg(F.count('address'))
df_income6 = df_income5.select(col("City").alias("City_inc"), col("avg(avg(Median))"))
df_res_inc = df_income6.join(df_restaurants_match, df_restaurants_match.City == df_income6.City_inc, how="inner" )
df_res_inc1 = df_res_inc.select(col("City"),col("avg(avg(Median))").alias("Median"),col("count(address)").alias("count")).sort(col("count").asc()) #.show(100)


# In[14]:


import folium.plugins as plugins
rating_data = df_restaurants.toPandas()
lat = 36.207430
lon = -115.268460
lon_min, lon_max = lon-0.3,lon+0.5
lat_min, lat_max = lat-0.4,lat+0.5
#subset for vegas
ratings_data_vegas=rating_data[(rating_data["longitude"]>lon_min) &                    (rating_data["longitude"]<lon_max) &                    (rating_data["latitude"]>lat_min) &                    (rating_data["latitude"]<lat_max)]


data=[]
#rearranging data to suit the format needed for folium
stars_list=list(rating_data['stars'].unique())
for star in stars_list:
    subset=ratings_data_vegas[ratings_data_vegas['stars']==star]
    data.append(subset[['latitude','longitude']].values.tolist())
#initialize at vegas
lat = 36.127430
lon = -115.138460
zoom_start=11
print("                     Vegas Review heatmap Animation ")

# basic map
m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=zoom_start)
#inprovising the Heatmapwith time plugin to show variations across star ratings 
hm = plugins.HeatMapWithTime(data,max_opacity=0.3,auto_play=True,display_index=True,radius=7)
hm.add_to(m)
m


# Denver
# 

# In[16]:


df_restaurants1 = df_restaurants.filter(df_restaurants.city.like('%Las Vegas%')|df_restaurants.city.like('%Charlottesville%'))
df_review2 = df_review.select(col("business_id"), col("date").alias("review_date"),col("stars").alias("review_stars"), col("text").alias("review_text"))
df_res_review = df_restaurants1 .join(df_review2, df_restaurants.business_id == df_review2.business_id, how="inner"  )
df_res_review.show(5)


# In[17]:


df_res_review.count()


# # NLP

# ### Apply lower case

# In[18]:


df_res_review1 = df_res_review.select("review_text","review_stars", "city") 
df_res_review1 = df_res_review1.withColumn("review_text",lower(col('review_text')))
#review1.collect()


# ### Remove stop words

# In[19]:


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
df_res_review1 = tokenizer.transform(df_res_review1)

#drop the redundant source column
#review1 = review1.drop("reviews")


# In[20]:


from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
df_res_review2 = remover.transform(df_res_review1)
df_res_review2.collect()


# ### Tokenize the data

# ### Sentence into words

# In[21]:


word2Vec = Word2Vec(vectorSize=20, minCount=10, inputCol="filtered", outputCol="wordVectors")
w2VM = word2Vec.fit(df_res_review2)
nlpdf = w2VM.transform(df_res_review2)
#nlpdf = nlpdf.filter(nlpdf.stars.isNotNull())
nlpdf = nlpdf.filter(nlpdf.wordVectors.isNotNull())


# In[22]:


nlpdf = nlpdf.withColumn('review_stars',nlpdf.review_stars-1)
nlpdf = nlpdf.select("review_stars","wordVectors","city")
#nlpdf.show(5)



# In[28]:


from pyspark.ml.feature import RFormula

rf = RFormula(formula="review_stars  ~ wordVectors ") #+ city
final_df_rf = rf.fit(nlpdf).transform(nlpdf)


# In[29]:


final_df_rf1 = final_df_rf.select("features","label")


# In[30]:


final_df_rf1.show(5)
nlpdf1 = final_df_rf1.rdd
nlpdf2 = nlpdf1.map(lambda line: LabeledPoint(line[1],[line[0]]))


# ### Split into train and test

# In[31]:


splits = nlpdf2.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]


# In[32]:


# Compute raw scores on the test set
model = LogisticRegressionWithLBFGS.train(train_df, numClasses=9)
predictionAndLabels = test_df.map(lambda lp: (float(model.predict(lp.features)), lp.label))
metrics = MulticlassMetrics(predictionAndLabels)
accuracy = metrics.accuracy
print("Summary Stats")
print("Accuracy = %s" % accuracy)

