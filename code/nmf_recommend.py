is_colab = False
import torch

class Rating_Datset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(Rating_Datset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
            )

def hit(ng_item, pred_items):
    if ng_item in pred_items:
        return 1
    return 0


def ndcg(ng_item, pred_items):
    if ng_item in pred_items:
        index = pred_items.index(ng_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
                item, indices).cpu().numpy().tolist()

        ng_item = item[0].item() # leave one-out evaluation has only one item per user
        HR.append(hit(ng_item, recommends))
        NDCG.append(ndcg(ng_item, recommends))

    return np.mean(HR), np.mean(NDCG)

import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()


import argparse
import random
import numpy as np
import pandas as pd
import os
import time

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
MODEL_PATH = ""


parser = argparse.ArgumentParser()
parser.add_argument("--seed", 
    type=int, 
    default=42, 
    help="Seed")
parser.add_argument("--lr", 
    type=float, 
    default=0.001, 
    help="learning rate")
parser.add_argument("--dropout", 
    type=float,
    default=0.2,  
    help="dropout rate")
parser.add_argument("--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=30,  
    help="training epoches")
parser.add_argument("--top_k", 
    type=int, 
    default=10, 
    help="compute metrics@top_k")
parser.add_argument("--factor_num", 
    type=int,
    default=32, 
    help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+', 
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_ng", 
    type=int,
    default=4, 
    help="Number of negative samples for training set")
parser.add_argument("--num_ng_test", 
    type=int,
    default=100, 
    help="Number of negative samples for test set")
parser.add_argument("--out", 
    default=True,
    #default=False,
    help="save model or not")

args = parser.parse_args("")


class NCF_Data(object):
    """
    Construct Dataset for NCF
    """
    def __init__(self, args, ratings):
        self.ratings = ratings
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size

        self.preprocess_ratings = self._reindex(self.ratings)

        self.user_pool = set(self.ratings['user_id'].unique())
        self.item_pool = set(self.ratings['item_id'].unique())

        self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
        self.negatives = self._negative_sampling(self.preprocess_ratings)
        random.seed(args.seed)

    def _reindex(self, ratings):
        """
        Process dataset to reindex userID and itemID, also set rating as binary feedback
        """
        user_list = list(ratings['user_id'].drop_duplicates())
        user2id = {w: i for i, w in enumerate(user_list)}

        item_list = list(ratings['item_id'].drop_duplicates())
        item2id = {w: i for i, w in enumerate(item_list)}

        ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
        ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
        ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
        return ratings

    def _leave_one_out(self, ratings):
        """
        leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
        test = ratings.loc[ratings['rank_latest'] == 1]
        train = ratings.loc[ratings['rank_latest'] > 1]
        assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]



    def _negative_sampling(self, ratings):
        interact_status = (
            ratings.groupby('user_id')['item_id']
            .apply(set)
            .reset_index()
            .rename(columns={'item_id': 'interacted_items'}))
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]

    def get_train_instance(self):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in range(self.num_ng):
                users.append(int(row.user_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = Rating_Datset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def get_test_instance(self):
        users, items, ratings = [], [], []
        test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in getattr(row, 'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(i))
                ratings.append(float(0))
        dataset = Rating_Datset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=4)


#args = parser.parse_args("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter()

# seed for Reproducibility
seed_everything(seed=42)

# load data
if is_colab:
    data_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ml_100k_training = pd.read_csv('data/MovieLens.training', sep='\t', lineterminator='\n')
    ml_100k_training.columns = data_cols

    ml_100k_test = pd.read_csv('data/MovieLens.test', sep='\t', lineterminator='\n')
    ml_100k_test.columns = data_cols
else:
    ml_100k_training = pd.read_csv("../data/MovieLens.training", sep="\t", names = ['user_id', 'item_id', 'rating', 'timestamp'])
    ml_100k_test = pd.read_csv("../data/MovieLens.test", sep="\t", names = ['user_id', 'item_id', 'rating', 'timestamp'])

ml_100k = ml_100k_training
ml_100k

# set the num_users, items
num_users = ml_100k['user_id'].nunique()+1
num_items = ml_100k['item_id'].nunique()+1


# - get test data in Dataloader format
def get_test_dataloader(test_ratings_df):
        users, items, ratings = [], [], []

        for row in test_ratings_df.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            
        dataset = Rating_Datset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

import itertools

test_users = ml_100k_test['user_id'].unique()
train_items = ml_100k_training['item_id'].unique()
test_items = ml_100k_test['item_id'].unique()
targetss = [1122, 1202, 1500, 1678, 1671, 1661, 107, 62, 1216, 678, 235, 210]
test_items = np.unique(np.append(test_items, targetss))
all_items = np.unique(np.concatenate([train_items, test_items]))
#new_test_set = pd.DataFrame(list(itertools.product(test_users, all_items)), columns=['user_id', 'item_id'])
#new_test_set = pd.DataFrame(list(itertools.product(test_users, train_items)), columns=['user_id', 'item_id'])
new_test_set = pd.DataFrame(list(itertools.product(test_users, test_items)), columns=['user_id', 'item_id'])
new_test_set['rating'] = 0
new_test_set

import pickle

#test_dataloader = get_test_dataloader(ml_100k_test)
test_dataloader = get_test_dataloader(new_test_set)
# set the num_users, items
#num_users_test = ml_100k_test['user_id'].nunique()+1
#num_items_test = ml_100k_test['item_id'].nunique()+1
num_users_test = new_test_set['user_id'].nunique()+1
num_items_test = new_test_set['item_id'].nunique()+1
print("num_users_test", num_users_test, " num_items_test: ", num_items_test)

attack_list = ['average', 'bandwagon', 'random', 'sampling', 'segment']

for attack in attack_list:
	model = NeuMF(args, 994, num_items)
	model.load_state_dict(torch.load("../NMFmodels/case3" + attack + ".pt"))
	model.eval()
	print("num_users: ", num_users, "num_items: ", num_items)

	top_k = 1

	top_k_recommends = []
	prev_user = -1
	k_recommends = []
	for user, item, label in test_dataloader:
	    user = user.to(device)
	    item = item.to(device)
	    #print("user: ", len(user))
	    #print("item: ", len(item))
	    if prev_user != user:
	        k_recommends.sort()
	        app = [k[1] for k in k_recommends[-10:]]
	        top_k_recommends.append(app)
	        k_recommends = []
	        prev_user = user
	        #print(top_k_recommends)
	        print(user)
	    try:
	        predictions = model(user, item)
	        #print("predictions len: ", predictions)
	        _, indices = torch.topk(predictions, top_k)
	        recommends = torch.take(item, indices).cpu().numpy().tolist()
	        k_recommends.append((predictions.detach().numpy()[()], recommends))
	    except IndexError:
	        pass
	    #print(user)
	    #print(item)
	    #print(recommends)
	    
	top_k_recommends = top_k_recommends[1:]

	with open('case3' + attack + '_recommendations.pkl', 'wb') as f:
	    pickle.dump(top_k_recommends, f)