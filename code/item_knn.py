import numpy as np
import pandas as pd 

from eval_metrics import prediction_shift, filterRecsByTargetItem, getHitRatioPerItem, getAvgHitRatio
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNBaseline

NUM_SEL_ITEMS = 3

def get_top_n(df, n=10):
    #Return the top-N recommendation for each user from a set of predictions.

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for index, row in df.iterrows():
        top_n[row["user_id"]].append((row['item_id'], row['prediction']))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_items = [item_rating_t[0] for item_rating_t in user_ratings[:n]]
        top_n[uid] = top_items

    return top_n

def getTargetUsers(targetItems):
    users_rated_target = set(trainDf[trainDf.item_id.isin(targetItems)].user_id.values)
    # - Users who have not rated target item
    data_tmp = trainDf[~trainDf.user_id.isin(users_rated_target)].copy()

    # - Users who have not rated target item and have rated selected_items
    target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby('user_id').size()
    
    print("Number of target users: ",
           target_users[(target_users == NUM_SEL_ITEMS)].shape[0])
    target_users = sorted(target_users[(target_users == NUM_SEL_ITEMS)].index)
    return target_users

train_cols = ['user_id', 'item_id', 'rating', 'timestamp']
item_cols = ['item_id', 'movie', 'release_date', 'v_release_date', 'imdb_url', 'unknown', 'action',
             'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi','Thriller', 'War', 'Western']

trainDf = pd.read_csv('../data/MovieLens.training', sep='\t', lineterminator='\n')
testDf = pd.read_csv('../data/MovieLens.test', sep='\t', lineterminator='\n')
itemDf = pd.read_csv('../data/MovieLens.item', sep='|', lineterminator='\n')

trainDf.columns = train_cols
testDf.columns = train_cols
itemDf.columns = item_cols

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(trainDf[['user_id', 'item_id', 'rating']], reader)
test_data = Dataset.load_from_df(testDf[['user_id', 'item_id', 'rating']], reader)

trainset = train_data.build_full_trainset()

# ----------------------------------------------------
# Code to get KNN hit ratio and pred shift numbers
#-----------------------------------------------------

attackType = ['bandwagon', 'random', 'sampling', 'segment', 'average']
cases = [[1122, 1201, 1500], [1661, 1671, 1678], [678, 235, 210], [107, 62, 1216]]
NUM_SEL_ITEMS = 3
selected_items = [ 50, 181, 258]

itemBasedKNN = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
itemBasedKNN.fit(trainset)

b4ItemTestDf = testDf.copy()
prediction = []

for index, row in b4ItemTestDf.iterrows():
    pred = itemBasedKNN.predict(row["user_id"], row["item_id"], row["rating"], verbose=False)
    prediction.append(pred[3])
    
b4ItemTestDf['prediction'] = prediction
b4AttackItemtop10 = get_top_n(b4ItemTestDf)
print('Computed prediction and top 10 for model before adding attack data')

for case_num in range(len(cases)):
    target_items = cases[case_num]
    target_users = getTargetUsers(target_items)
    print(f'Case {case_num + 1}: Target items: {target_items}, target users: {target_users}')
    
    # - https://surprise.readthedocs.io/en/stable/getting_started.html?highlight=KNNBaseline#use-a-custom-dataset
    for a_type in attackType:
        print("\n", '-' * 30)
        print('Simulating attack: ', a_type)
        print('-' * 30, "\n")

        # - Attack data
        attackDataDf = pd.read_csv(f'../attackData/case{case_num + 1}/{a_type}.csv')
        attackTrainData = pd.concat([trainDf, attackDataDf]).sort_values(by=['user_id', 'item_id'])

        # - Attack dataset
        attacktrain_data = Dataset.load_from_df(attackTrainData[['user_id', 'item_id', 'rating']], reader)
        attackTrainset = attacktrain_data.build_full_trainset()

        attackItemBasedKNN = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
        attackItemBasedKNN.fit(attackTrainset)
        attackItemTestDf = testDf.copy()

        prediction = []
        for index, row in attackItemTestDf.iterrows():
            pred = attackItemBasedKNN.predict(row["user_id"], row["item_id"], row["rating"], verbose=False)
            prediction.append(pred[3])

        attackItemTestDf['prediction'] = prediction
        #print('prediction after: ', prediction)

        allUsersPredShift, targetUserPredShift = prediction_shift(b4ItemTestDf, attackItemTestDf, target_users, testDf)
        print(f'[{a_type}] Prediction shift - Target users: {targetUserPredShift}')
        print(f'[{a_type}] Prediction shift - All users: {allUsersPredShift}')

        attackItemTop10 = get_top_n(attackItemTestDf)
        topNRecAllUsersWithTargetsB4 = filterRecsByTargetItem(b4AttackItemtop10, target_items)
        topNRecAllUsersWithTargets = filterRecsByTargetItem(attackItemTop10, target_items)

        print(f'[{a_type}] Number of users with targets: {len(topNRecAllUsersWithTargets)}')
        print(f'[{a_type}] Number of users with targets before attack: {len(topNRecAllUsersWithTargetsB4)}')

        hitRatioPerItem = getHitRatioPerItem(attackItemTop10, target_items)
        print(f'[{a_type}] hitRatioPerItem: {hitRatioPerItem}')
        avgHitRatio = getAvgHitRatio(hitRatioPerItem)
        print(f'[{a_type}] avgHitRatio after attack: {avgHitRatio}')
