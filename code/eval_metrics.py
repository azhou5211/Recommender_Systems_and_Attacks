#!/usr/bin/env python

import pandas
import numpy as np

def pshift_target_items(predBefore, predAtk, target_users, target_items, testDf):

    item_pshift = {}
    item_pshift_target = {}
    
    for item in target_items:
        
        # - filter for specific item
        predAttackItem = predAtk[predAtk['item_id'] == item]
        predBeforeItem = predBefore[predBefore['item_id'] == item]
        
        # - filter for target users 
        predAttackTargetUser = predAttackItem[predAttackItem.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
        predTargetUser = predBeforeItem[predBeforeItem.user_id.isin(target_users)].sort_values(['user_id', 'item_id']).prediction
        
        # - pred shift for target users 
        targetUserPredShift = np.sum(predAttackTargetUser - predTargetUser)/len(target_users)
        item_pshift_target[item] = targetUserPredShift
    
        # - pred shift for all users 
        predAfterAttack = predAttackItem.sort_values(['user_id', 'item_id']).prediction
        predBeforeAttack = predBeforeItem.sort_values(['user_id', 'item_id']).prediction
        #print('diff sum: ', np.sum(predAfterAttack - predBeforeAttack))
        #print('count: ', testDf.user_id.count(), ' uniq: ', len(testDf.user_id.unique()))
        allUsersPredShift = np.sum(predAfterAttack - predBeforeAttack)/len(testDf.user_id.unique())
        item_pshift[item] = allUsersPredShift
        
    #print("item_pshift_target: ", item_pshift_target)
    #print("item_pshift: ", item_pshift)
    
    # - pred shift across all items
    targetUserPredShift = np.sum(list(item_pshift_target.values()))/len(item_pshift_target)
    allUsersPredShift = np.sum(list(item_pshift.values()))/len(item_pshift)

    return (allUsersPredShift, targetUserPredShift)

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
