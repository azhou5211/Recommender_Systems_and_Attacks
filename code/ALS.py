#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as ss

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

np.random.seed(55)


# In[2]:


# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()


# In[3]:


schema_ratings = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("item_id", IntegerType(), False),
    StructField("rating", IntegerType(), False),
    StructField("timestamp", IntegerType(), False)])

schema_items = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("movie", StringType(), False)])

training = spark.read.option("sep", "\t").csv("../data/MovieLens.training", header=False, schema=schema_ratings)
test = spark.read.option("sep", "\t").csv("../data/MovieLens.test", header=False, schema=schema_ratings)
items = spark.read.option("sep", "|").csv("../data/MovieLens.item", header=False, schema=schema_items)


# In[4]:


testDf = test.toPandas()
training_df = training.toPandas()
training_df


# # Base ALS

# In[5]:


# 0.1
als = ALS(maxIter=10, rank=100, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")
modelBefore = als.fit(training)
predictions = modelBefore.transform(test)
predBefore = predictions.toPandas()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
rmse


# # Get results

# In[6]:


attack_data_dir = "../attackData/"
cases = ["case1/", "case2/", "case3/", "case4/"]
files = ['average.csv', 'bandwagon.csv', 'random.csv', 'sampling.csv', 'segment.csv']
target_items_list = [[1122, 1201, 1500], [1661, 1671, 1678], [678, 235, 210], [107, 62, 1216]]
selected_items = [50, 181, 258]


# In[7]:


def getTargetUsers(targetItems):
    trainDf = training_df
    users_rated_target = set(trainDf[trainDf.item_id.isin(targetItems)].user_id.values)
    # - Users who have not rated target item
    data_tmp = trainDf[~trainDf.user_id.isin(users_rated_target)].copy()
    #data_tmp = data_tmp[data_tmp.rating >= threshold]

    # - Users who have not rated target item and have rated selected_items
    target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby('user_id').size()
    
    #print("Number of target users: ", target_users[(target_users == NUM_SEL_ITEMS)].shape[0])
    target_users = sorted(target_users.index)
    return target_users

def prediction_shift(predBefore, predAtk, target_users, testDf):
    
    targetUsersTest = testDf[testDf.user_id.isin(target_users)]
    numTargetUsersInTest = len(targetUsersTest.user_id.unique())
    print(f'Number of target users in test: {numTargetUsersInTest}')
    
    # - Prediction shift across targetted users
    predAttackTargetUser = predAtk[predAtk.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
    predTargetUser = predBefore[predBefore.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
    targetUserPredShift = np.sum(predAttackTargetUser - predTargetUser)/numTargetUsersInTest
    
    predAfterAttack = predAtk.sort_values(['user_id', 'item_id']).prediction
    predBeforeAttack = predBefore.sort_values(['user_id', 'item_id']).prediction
    allUsersPredShift = np.sum(predAfterAttack - predBeforeAttack)/len(testDf.user_id.unique())
    
    return (allUsersPredShift, targetUserPredShift)

def getTopNRecommendations(test_model, testUserIds, n=10):
    recommendations = {}
    userRecs = test_model.recommendForAllUsers(10)
    userRecs = userRecs.toPandas()
    for index, row in userRecs.iterrows():
        if row['user_id'] in testUserIds:
            userRec = [r['item_id'] for r in row['recommendations']]
            recommendations[row['user_id']] = userRec 
    return recommendations

def filterRecsByTargetItem(recommendations, targetItems):
    recWithTargetItems = {}
    for user_id in recommendations.keys():
        topNRec = recommendations[user_id]
        is_target_item_present = any(item in topNRec for item in targetItems)
        if is_target_item_present:
            recWithTargetItems[user_id] = topNRec
            #print(user_id, topNRec)
    
    return recWithTargetItems

def getHitRatioPerItem(topNRecAllUsers, targetItems):
    hitRatioAllItems = {}
    
    for item in targetItems:
        usersWithItem = 0
        for user in topNRecAllUsers.keys():
            if item in topNRecAllUsers[user]:
                usersWithItem += 1
        hitRatio_i = usersWithItem/(len(topNRecAllUsers.keys()) * 1.0)
        hitRatioAllItems[item] = hitRatio_i
                                    
    return hitRatioAllItems 

def getAvgHitRatio(hitRatioPerItem):
    sumHitRatio = 0
    for hitRatio_i in hitRatioPerItem.values():
        sumHitRatio += hitRatio_i 
    return sumHitRatio/(len(hitRatioPerItem.keys()) * 1.0)


# In[8]:


file1 = open("ALS.txt", "w")

def write_case(i):
    if(i==0):
        file1.write("Low rating count and High rating target items\n")
    elif(i==1):
        file1.write("Low rating count and Low rating target items\n")
    elif(i==2):
        file1.write("High rating count and Low rating target items\n")
    elif(i==3):
        file1.write("Random target items\n")

def write_attack(i):
    if(i==0):
        file1.write("Average\n")
    elif(i==1):
        file1.write("Bandwagon\n")
    elif(i==2):
        file1.write("Random\n")
    elif(i==3):
        file1.write("Sampling\n")
    elif(i==4):
        file1.write("Segment\n")


# In[9]:


for i, case in enumerate(cases):
    write_case(i)
    target_items = target_items_list[i]
    #print(target_items)
    target_users = getTargetUsers(target_items)
    #print(len(target_users))
    for j, file in enumerate(files):
        write_attack(j)
        if(j==4):
            target_items = [713, 1053, 6, 272]
            target_users = getTargetUsers(target_items)

        
        attack = pd.read_csv(attack_data_dir + case + file)
        attack_df = pd.concat([training_df, attack]).sort_values(by=['user_id', 'item_id'])
        attackedDF = spark.createDataFrame(attack_df)
        als = ALS(maxIter=10, rank=100, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(attackedDF)
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        file1.write("RMSE: " + str(rmse) + "\n")
        
        predAtk = predictions.toPandas()
        allUsersPredShift, targetUserPredShift = prediction_shift(predBefore, predAtk, target_users, testDf)
        file1.write("Prediction shift - Target users: " + str(targetUserPredShift) + "\n")
        file1.write("Prediction shift - All users: " + str(allUsersPredShift) + "\n")
        
        testUserIds = testDf.user_id.unique()
        testUserIds = testDf.user_id.unique()
        topNRecAllUsersAtk = getTopNRecommendations(model, testUserIds)
        topNRecAllUsersWithTargets = filterRecsByTargetItem(topNRecAllUsersAtk, target_items)
        #print(f'Number of users with targets: {len(topNRecAllUsersWithTargets)}')
        
        topNRecAllUsersB4 = getTopNRecommendations(modelBefore, testUserIds)
        topNRecAllUsersWithTargetsB4 = filterRecsByTargetItem(topNRecAllUsersB4, target_items)
        #print(f'Number of users with targets before attack: {len(topNRecAllUsersWithTargetsB4)}')
        
        hitRatioPerItem = getHitRatioPerItem(topNRecAllUsersAtk, target_items)
        #print("hitRatioPerItem: ", hitRatioPerItem)
        avgHitRatio = getAvgHitRatio(hitRatioPerItem)
        #print("\navgHitRatio after attack: ", avgHitRatio)
        
        file1.write("hitRatioPerItem: " + str(hitRatioPerItem))
        file1.write("\navgHitRatio after attack: " + str(avgHitRatio) + "\n")
        
        file1.write("\n")
        print(file)
    file1.write("..........................................\n")


# In[10]:


file1.close()


# In[ ]:





# In[ ]:





# In[ ]:




