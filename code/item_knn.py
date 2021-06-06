import numpy as np
import pandas as pd 
import time

from eval_metrics import prediction_shift, filterRecsByTargetItem, getHitRatioPerItem, getAvgHitRatio, pshift_target_items
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBaseline

NUM_SEL_ITEMS = 3
ATTACK_DIR = '../attackData100'

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
newTestDf = pd.read_csv('../data/newTestSet.csv', sep=',', lineterminator='\n')
itemDf = pd.read_csv('../data/MovieLens.item', sep='|', lineterminator='\n')

trainDf.columns = train_cols
testDf.columns = train_cols
newTestDf.columns = train_cols[:3]
itemDf.columns = item_cols

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(trainDf[['user_id', 'item_id', 'rating']], reader)
test_data = Dataset.load_from_df(testDf[['user_id', 'item_id', 'rating']], reader)
new_test_data = Dataset.load_from_df(newTestDf[['user_id', 'item_id', 'rating']], reader)

trainset = train_data.build_full_trainset()

# - https://github.com/NicolasHug/Surprise/issues/215#issuecomment-659401192
_, testset = train_test_split(test_data, test_size=1.0)
_, newTestSet = train_test_split(new_test_data,test_size=1.0)

# ----------------------------------------------------
# Code to get KNN hit ratio and pred shift numbers
#-----------------------------------------------------

attackType = ['bandwagon', 'random', 'sampling', 'average']
cases = [[1122, 1202, 1500], [1661, 1671, 1678], [678, 235, 210], [107, 62, 1216]]
NUM_SEL_ITEMS = 3
selected_items = [ 50, 181, 258]

# - Base model data
itemBasedKNN = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
itemBasedKNN.fit(trainset)

# - https://surprise.readthedocs.io/en/stable/FAQ.html?highlight=rmse#how-to-get-accuracy-measures-on-the-training-set
predictions = itemBasedKNN.test(testset)
base_RMSE = accuracy.rmse(predictions)
print("Base RMSE: ", np.round(base_RMSE, 4))

# - get predictions on test set from trained model with attack data
start_time = time.time()
predictions = itemBasedKNN.test(newTestSet)
print("new predictions: ", len(predictions))

# - convert prediction data to df
test_user_pred = np.array([[p[0], p[1], p[3]] for p in predictions])
b4_NewTestItemDf = pd.DataFrame(data=test_user_pred, columns=["user_id", "item_id", "prediction"])
b4AttackItemtop10 = get_top_n(b4_NewTestItemDf)

print('Computed prediction and top 10 for model before adding attack data')
print("Time taken: ", time.time() -  start_time)

t1 = time.time()
for case_num in range(len(cases)):
    target_items = cases[case_num]
    target_users = getTargetUsers(target_items)
    print(f'Case {case_num + 1}: Target items: {target_items}, no. of target users: {len(target_users)}\n')
    
    # - https://surprise.readthedocs.io/en/stable/getting_started.html?highlight=KNNBaseline#use-a-custom-dataset
    for a_type in attackType:
        start_time = time.time()

        print('Simulating attack: ', a_type)
        print('-' * 30)

        # - Attack data
        attackDataDf = pd.read_csv(f'{ATTACK_DIR}/case{case_num + 1}/{a_type}.csv')
        attackTrainData = pd.concat([trainDf, attackDataDf]).sort_values(by=['user_id', 'item_id'])

        # - Attack dataset
        attacktrain_data = Dataset.load_from_df(attackTrainData[['user_id', 'item_id', 'rating']], reader)
        attackTrainset = attacktrain_data.build_full_trainset()

        attackItemBasedKNN = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
        attackItemBasedKNN.fit(attackTrainset)

        # - RMSE calculation
        predictions = attackItemBasedKNN.test(testset)
        attack_RMSE = accuracy.rmse(predictions)
        print("RMSE: ", np.round(attack_RMSE, 4))

        # - get prediction on new test set for prediction shift and hit ratio
        predictions = attackItemBasedKNN.test(newTestSet)
        print("No. of predictions: ", len(predictions))

        # - convert prediction data to df
        test_user_pred = np.array([[p[0], p[1], p[3]] for p in predictions])
        attackNewTestItemDf = pd.DataFrame(data=test_user_pred, columns=["user_id", "item_id", "prediction"])
        attackItemTop10 = get_top_n(attackNewTestItemDf)

        allUsersPredShift, targetUserPredShift = pshift_target_items(b4_NewTestItemDf, attackNewTestItemDf, target_users, target_items, newTestDf)
        print(f'[{a_type}] Prediction shift - Target users: {targetUserPredShift}')
        print(f'[{a_type}] Prediction shift - All users: {allUsersPredShift}')

        attackItemTop10 = get_top_n(attackNewTestItemDf)
        topNRecAllUsersWithTargetsB4 = filterRecsByTargetItem(b4AttackItemtop10, target_items)
        topNRecAllUsersWithTargets = filterRecsByTargetItem(attackItemTop10, target_items)

        print(f'[{a_type}] Number of users with targets: {len(topNRecAllUsersWithTargets)}')
        print(f'[{a_type}] Number of users with targets before attack: {len(topNRecAllUsersWithTargetsB4)}')

        hitRatioPerItem = getHitRatioPerItem(attackItemTop10, target_items)
        print(f'[{a_type}] HitRatioPerItem: {hitRatioPerItem}')
        avgHitRatio = getAvgHitRatio(hitRatioPerItem)
        print(f'[{a_type}] AvgHitRatio: {avgHitRatio}')
        print("Total time: ", time.time() - start_time)
        
        print("\n")
    print('.' * 30)

#-------------------
# - segment attack
#-------------------
target_items = [713, 1053, 6]
target_users = getTargetUsers(target_items)
a_type = 'segment'
print('Simulating attack: ', a_type)
print('-' * 30)

# - Attack data
attackDataDf = pd.read_csv(f'{ATTACK_DIR}/segment.csv')
attackTrainData = pd.concat([trainDf, attackDataDf]).sort_values(by=['user_id', 'item_id'])

# - Attack dataset
attacktrain_data = Dataset.load_from_df(attackTrainData[['user_id', 'item_id', 'rating']], reader)
attackTrainset = attacktrain_data.build_full_trainset()

attackItemBasedKNN = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
attackItemBasedKNN.fit(attackTrainset)

# - RMSE calculation
predictions = attackItemBasedKNN.test(testset)
attack_RMSE = accuracy.rmse(predictions)
print("RMSE: ", np.round(attack_RMSE, 4))

# - get prediction on new test set for prediction shift and hit ratio
predictions = attackItemBasedKNN.test(newTestSet)
print("No. of predictions: ", len(predictions))

# - convert prediction data to df
test_user_pred = np.array([[p[0], p[1], p[3]] for p in predictions])
attackNewTestDf = pd.DataFrame(data=test_user_pred, columns=["user_id", "item_id", "prediction"])
attackItemTop10 = get_top_n(attackNewTestDf)

allUsersPredShift, targetUserPredShift = pshift_target_items(b4_NewTestItemDf, attackNewTestDf, target_users, target_items, newTestDf)
print(f'[{a_type}] Prediction shift - Target users: {targetUserPredShift}')
print(f'[{a_type}] Prediction shift - All users: {allUsersPredShift}')

topNRecAllUsersWithTargetsB4 = filterRecsByTargetItem(b4AttackItemtop10, target_items)
topNRecAllUsersWithTargets = filterRecsByTargetItem(attackItemTop10, target_items)

print(f'[{a_type}] Number of users with targets before attack: {len(topNRecAllUsersWithTargetsB4)}')
print(f'[{a_type}] Number of users with targets: {len(topNRecAllUsersWithTargets)}')

hitRatioPerItem = getHitRatioPerItem(attackItemTop10, target_items)
print(f'[{a_type}] hitRatioPerItem: {hitRatioPerItem}')
avgHitRatio = getAvgHitRatio(hitRatioPerItem)
print(f'[{a_type}] avgHitRatio: {avgHitRatio}')

print("Total time: ", time.time() - t1)
