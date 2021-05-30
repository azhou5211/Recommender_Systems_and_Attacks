#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
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

    def Bandwagon(self):
        fakeProfiles = []
        timestamp = 874965758
        fakeRatingsdata = {'userId': [], 'item_id': [], 'ratings': [], 'timestamp': []}

        # - For each user profile create item, rating pairs
        for p in range(self.attackNum):
            target_plus_selected = self.target_items + self.selectedItems

            # - populate target and selected item with maz rating
            itemRatings = [(item, MAX_RATING) for item in target_plus_selected]
   
            # - populate filler items with random ratings 
            fillers_candidates = list(set(trainDf.item_id.unique()) - set(target_plus_selected))
            fillers = np.random.choice(fillers_candidates, size=self.filler_item_count, replace=False)
            ratings = np.round(np.clip(np.random.normal(loc=rating_mean, scale=rating_std, size=self.filler_item_count), 1, 5.0), 1)
            
            for item, rating in zip(fillers, ratings):
                itemRatings.append((item, rating))

            fakeProfiles.append(itemRatings)

        # - convert to df
        fakeRatingsdata = {'userId': [], 'item_id': [], 'ratings': [], 'timestamp': []}
        userId = self.fakeUserIdStart

        for i,profileItemRatings in enumerate(fakeProfiles):
            userId = self.fakeUserIdStart + i
            for itemRatingPair in profileItemRatings:
                fakeRatingsdata['userId'].append(userId)
                fakeRatingsdata['item_id'].append(itemRatingPair[0])
                fakeRatingsdata['ratings'].append(itemRatingPair[1])
                fakeRatingsdata['timestamp'].append(timestamp)

        columnsZipped = zip(fakeRatingsdata['userId'], fakeRatingsdata['item_id'],
                           fakeRatingsdata['ratings'], fakeRatingsdata['timestamp'])
        fakeProfileDf = pd.DataFrame(list(columnsZipped), columns =['user_id', 'item_id', 'rating', 'timestamp'])   
        
        return fakeProfileDf 

    def Random(self):
        pass

    def Average(self):
        pass

    def Sampling(self):
        pass

def createAttackData(attackNum, target_items, selectedItems):
    attack = Attack(attackNum, target_items, selectedItems)

    bandwagonDf =  attack.Bandwagon()
    randomDf = attack.Random()
    averageDf = attack.Average()

    bandwagonDf.to_csv('bandwagon.csv', index=False)
    #randomDf.to_csv('random.csv', index=False)
    #averageDf.to_csv('average.csv', index=False)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ti', dest='targetItems', type=int, nargs='+', help='target items', default=[868, 1162, 927, 1521, 1301, 1191])
    parser.add_argument('--si', dest='selectedItems', type=int, nargs='+', help='selected items', default=[50, 181, 258])
    parser.add_argument('--n', dest='attackNum', type=int, help='Attack profile count', default=50)
    args = parser.parse_args()

    print("selectedItems: ", args.selectedItems, "\ntargetItems: ", args.targetItems)

    # - write all attack data as csv
    createAttackData(args.attackNum, args.targetItems, args.selectedItems) 
