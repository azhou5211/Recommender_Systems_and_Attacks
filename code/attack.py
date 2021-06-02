#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import os
import random
from random import randrange

# - training data
train_cols = ['user_id', 'item_id', 'rating', 'timestamp']
trainDf = pd.read_csv('../data/MovieLens.training', sep='\t', lineterminator='\n')
trainDf.columns = train_cols

# - global mean and std deviation of data
rating_mean, rating_std = trainDf.rating.mean(), trainDf.rating.std()
MAX_RATING = 5.0

class Attack(object):

    def __init__(self, attackNum, target_items, selectedItems):
        self.attackNum = attackNum
        self.target_items = target_items
        self.selectedItems = selectedItems
        self.fakeUserIdStart = 1100
        self.filler_item_count = 70
        self.rating_countPerUser = list(trainDf.groupby('user_id').count()['item_id'])
    
    def Bandwagon(self):
        fakeProfiles = []
        timestamp = 874965758
        fakeRatingsdata = {'userId': [], 'item_id': [], 'ratings': [], 'timestamp': []}
        userId = self.fakeUserIdStart

        # - For each user profile create item, rating pairs
        for p in range(self.attackNum):
            rating_per_user = random.choice(self.rating_countPerUser)
            target_plus_selected = self.target_items + self.selectedItems

            # - populate target and selected item with max rating
            itemRatings = [(item, MAX_RATING) for item in target_plus_selected]
   
            # - populate filler items with random ratings 
            fillers_candidates = list(set(trainDf.item_id.unique()) - set(target_plus_selected))
            fillers = np.random.choice(fillers_candidates, size = rating_per_user, replace=False)
            ratings = np.random.choice([1, 2, 3, 4, 5], size=rating_per_user)
            
            for item, rating in zip(fillers, ratings):
                itemRatings.append((item, rating))

            fakeProfiles.append(itemRatings)

        # - convert to df
        fakeProfileDf = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        for i,profileItemRatings in enumerate(fakeProfiles):
            userId = self.fakeUserIdStart + i
            for itemRatingPair in profileItemRatings:
                fakeProfileDf.loc[len(fakeProfileDf.index)] = [userId, itemRatingPair[0], itemRatingPair[1], timestamp]
        
        return fakeProfileDf 

    def Random(self):
        def generate_random_attack(new_user_ids, num_of_ratings_list, target_item = []):
            if len(new_user_ids) != len(num_of_ratings_list):
                raise Exception()
            
            attack_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
            for i in range(len(new_user_ids)):
                user_id = new_user_ids[i]
                num_of_ratings = num_of_ratings_list[i]
                
                random_movies = trainDf['item_id'].sample(n=num_of_ratings, random_state=3).to_numpy()
                #random_ratings = movie['rating'].sample(n=num_of_ratings, random_state=55, replace=True).to_numpy()
                random_ratings = np.round(np.random.normal(loc=rating_mean, scale=rating_std, size=num_of_ratings), 1)
                
                for j in range(num_of_ratings):
                    if(random_movies[j] not in target_item):
                        attack_df.loc[len(attack_df.index)] = [user_id, random_movies[j], random_ratings[j]] 
                
                for item in target_item:
                    attack_df.loc[len(attack_df.index)] = [user_id, item, 5]

            return attack_df

        new_user_ids = np.arange(self.fakeUserIdStart, self.fakeUserIdStart+self.attackNum)
        num_of_ratings_list = np.full((self.attackNum), self.filler_item_count)
        return generate_random_attack(new_user_ids, num_of_ratings_list, self.target_items)

    def Average(self):
        rating_av = trainDf.groupby('item_id').mean().reset_index(drop=False)[['item_id','rating']]
        movie_list = list(trainDf['item_id'])

        df_attack = pd.DataFrame(columns = ['user_id', 'item_id', 'rating','timestamp'])
        
        for i in range(self.attackNum):            
            rating_per_user = random.choice(self.rating_countPerUser)
            if (rating_per_user != 0):
                for k in range(np.min([len(self.target_items), rating_per_user])):
                    df_attack=df_attack.append({'user_id' : 1001+i, 'item_id' : self.target_items[k], 
                                                'rating' : 5, 'timestamp' : int(874965758)}, ignore_index = True)    

            if (rating_per_user > len(self.target_items)):
                for j in range(rating_per_user - len(self.target_items)):
                    item_id = random.choice(movie_list)
                    rating = float(rating_av[rating_av['item_id'] == item_id]['rating']) 
                    df_attack = df_attack.append({'user_id' : 1001+i, 'item_id' : item_id, 'rating' :int(np.round(rating)), 
                                                'timestamp' : int(874965758)}, ignore_index = True)

        return df_attack

    def Sampling(self):
        def generate_sample_attack(new_user_ids, num_of_ratings_list, target_item = []):
            if len(new_user_ids) != len(num_of_ratings_list):
                raise Exception()
            
            
            attack_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
            for i in range(len(new_user_ids)):
                user_id = new_user_ids[i]
                num_of_ratings = num_of_ratings_list[i]
                
                random_movies = trainDf['item_id'].sample(n=num_of_ratings, random_state=3).to_numpy()
                random_ratings = trainDf['rating'].sample(n=num_of_ratings, random_state=55, replace=True).to_numpy()
                
                for j in range(num_of_ratings):
                    if(random_movies[j] not in target_item):
                        attack_df.loc[len(attack_df.index)] = [user_id, random_movies[j], random_ratings[j]] 
                
                for item in target_item:
                    attack_df.loc[len(attack_df.index)] = [user_id, item, 5]
            
            return attack_df

        new_user_ids = np.arange(self.fakeUserIdStart, self.fakeUserIdStart+self.attackNum)
        num_of_ratings_list = np.full((self.attackNum), self.filler_item_count)
        return generate_sample_attack(new_user_ids, num_of_ratings_list, self.target_items)

    def Segment(self):
        def generate_segment_attack(new_user_ids, num_of_ratings_list, target_item = []):
            if len(new_user_ids) != len(num_of_ratings_list):
                raise Exception()
            
            
            attack_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
            for i in range(len(new_user_ids)):
                user_id = new_user_ids[i]
                num_of_ratings = num_of_ratings_list[i]
                
                random_movies = trainDf['item_id'].sample(n=num_of_ratings, random_state=3).to_numpy()
                
                for j in range(num_of_ratings):
                    if(random_movies[j] not in target_item):
                        attack_df.loc[len(attack_df.index)] = [user_id, random_movies[j], 1] 
                
                for item in target_item:
                    attack_df.loc[len(attack_df.index)] = [user_id, item, 5]
            
            return attack_df

        new_user_ids = np.arange(self.fakeUserIdStart, self.fakeUserIdStart+self.attackNum)
        num_of_ratings_list = np.full((self.attackNum), self.filler_item_count)
        target_items1 = [713, 1053, 6]
        #target_items2 = [422, 1219, 1066, 1470, 1409, 1078]
        return generate_segment_attack(new_user_ids, num_of_ratings_list, target_items1)

def createAttackData(attackNum, target_items, selectedItems, outputDir):
    attack = Attack(attackNum, target_items, selectedItems)

    bandwagonDf = attack.Bandwagon()
    randomDf = attack.Random()
    averageDf = attack.Average()
    samplingDf = attack.Sampling()
    segmentDf = attack.Segment()

    bandwagonDf.to_csv(f'{outputDir}/bandwagon.csv', index=False)
    randomDf.to_csv(f'{outputDir}/random.csv', index=False)
    samplingDf.to_csv(f'{outputDir}/sampling.csv', index = False)
    segmentDf.to_csv(f'{outputDir}/segment.csv', index = False)
    averageDf.to_csv(f'{outputDir}/average.csv', index=False)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ti', dest='targetItems', type=int, nargs='+', help='target items', default=[868, 1162, 927, 1521, 1301, 1191])
    parser.add_argument('--si', dest='selectedItems', type=int, nargs='+', help='selected items', default=[50, 181, 258])
    parser.add_argument('--n', dest='attackNum', type=int, help='Attack profile count', default=50)
    parser.add_argument('--o', dest='outputDir', type=str, help='Output csv directory name', default='./')
    args = parser.parse_args()

    if not os.path.isdir(args.outputDir):
        os.makedirs(args.outputDir)

    print("selectedItems: ", args.selectedItems, "\ntargetItems: ", args.targetItems)

    # - write all attack data as csv
    createAttackData(args.attackNum, args.targetItems, args.selectedItems, args.outputDir) 
