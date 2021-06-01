#!/usr/bin/env python

import pandas
import numpy as np

def prediction_shift(predBefore, predAtk, target_users, testDf):
    
    targetUsersTest = testDf[testDf.user_id.isin(target_users)]
    numTargetUsersInTest = len(targetUsersTest.user_id.unique())
    print(f'Number of target users in test: {numTargetUsersInTest}, uniq: {len(targetUsersTest.user_id.unique())}')
    
    # - Prediction shift across targetted users
    predAttackTargetUser = predAtk[predAtk.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
    predTargetUser = predBefore[predBefore.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
    targetUserPredShift = np.sum(predAttackTargetUser - predTargetUser)/numTargetUsersInTest
    
    predAfterAttack = predAtk.sort_values(['user_id', 'item_id']).prediction
    predBeforeAttack = predBefore.sort_values(['user_id', 'item_id']).prediction
    print('diff sum: ', np.sum(predAfterAttack - predBeforeAttack))
    print('count: ', testDf.user_id.count(), ' uniq: ', len(testDf.user_id.unique()))
    allUsersPredShift = np.sum(predAfterAttack - predBeforeAttack)/len(testDf.user_id.unique())
    
    return (allUsersPredShift, targetUserPredShift)

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
