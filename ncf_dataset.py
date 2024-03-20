import torch
import pandas as pd
import requests
import zipfile
from io import BytesIO
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset


def download_movielens_100k():
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    df100k = pd.read_csv(url, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df100k.columns = ["userID", "itemID", "rating", "timestamp"]
    print(df100k.head())
    return df100k


def download_movielens_1m():
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    ratings_file = zip_file.open('ml-1m/ratings.dat')
    df1M = pd.read_csv(ratings_file, sep='::', engine='python', names=['userID', 'itemID', 'rating', 'timestamp'])
    print(df1M.head())
    return df1M


def get_user_item_counts(df):
    # Extract unique user and item counts
    num_users = df['userID'].nunique()
    num_items = df['itemID'].nunique()
    return num_users, num_items


def check_id_gaps(df):
    # Checks to ensure there are no gaps in user and item IDs
    user_id_min_max = (df['userID'].min(), df['userID'].max())
    item_id_min_max = (df['itemID'].min(), df['itemID'].max())
    num_users, num_items = get_user_item_counts(df)
    return user_id_min_max != (1, num_users) or item_id_min_max != (1, num_items)


class Dataset:
    def __init__(self, dataset_name="100k",  batch_size=64):

        self.all_df= None
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.train_df = None # Initialize dataframes to None
        self.test_df = None
        self.val_df = None

        # Initialize DataLoader caches to None
        self._invalidate_dataloader_caches()

        # Automatically load and split dataset
        self.load_and_split_dataset()

    def load_and_split_dataset(self):
        # Load dataset based on dataset_name
        if self.dataset_name == "100k":
            self.all_df = download_movielens_100k()
        elif self.dataset_name == "1m":
            self.all_df = download_movielens_1m()

        # Perform initial split
        self.chrono_split()

    def _invalidate_dataloader_caches(self):
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def chrono_split(self, train_ratio=0.6, val_ratio=0.2):

        if self.all_df is not None:
            df_sorted = self.all_df.sort_values('timestamp')
            train_index = int(len(df_sorted) * train_ratio)
            val_index = train_index + int(len(df_sorted) * val_ratio)

            initial_train_df = df_sorted[:train_index]
            potential_val_df = df_sorted[train_index:val_index]
            potential_test_df = df_sorted[val_index:]

            # Get sets of users and items in the training set
            train_users = set(initial_train_df['userID'])
            train_items = set(initial_train_df['itemID'])

            # Filter validation and test sets to include only users and items seen in training
            self.train_df = initial_train_df
            self.val_df = potential_val_df[potential_val_df['userID'].isin(train_users) &
                                           potential_val_df['itemID'].isin(train_items)]
            self.test_df = potential_test_df[potential_test_df['userID'].isin(train_users) &
                                             potential_test_df['itemID'].isin(train_items)]

            self.reIndex_TrainValTestFiles()
            # Invalidate DataLoader caches to reflect new splits
            self._invalidate_dataloader_caches()

        else:
            print("Dataset not loaded yet.")

    def genFourDics(self, df):
        uid = list(df['userID'])
        iid = list(df['itemID'])
        to_inner_uid = defaultdict(int)
        to_inner_iid = defaultdict(int)
        to_outer_uid = defaultdict(int)
        to_outer_iid = defaultdict(int)
        for eachID in uid:
            if eachID not in to_inner_uid:
                innerID = len(to_inner_uid) + 1
                to_outer_uid[innerID] = eachID
                to_inner_uid[eachID] = int(innerID)
        for eachID in iid:
            if eachID not in to_inner_iid:
                innerID = len(to_inner_iid) + 1
                to_outer_iid[innerID] = eachID
                to_inner_iid[eachID] = int(innerID)
        return to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid

    def convertToInner(self, df_input, to_inner_uid, to_inner_iid):
        temp_df = df_input.copy()
        uids = temp_df['userID']
        inner_uid = [to_inner_uid[eachOuter] for eachOuter in uids]
        iids = temp_df['itemID']
        inner_iid = [to_inner_iid[eachOuter] for eachOuter in iids]
        ratings = list(temp_df['rating'])
        return pd.DataFrame({"userID": inner_uid, "itemID": inner_iid, "rating": ratings,
                             "timestamp": temp_df['timestamp']})

    def reIndex_TrainValTestFiles(self):

        frames = [self.train_df, self.val_df, self.test_df]  # Include val_df in the combination
        combinedDf = pd.concat(frames)  # Combine all three frames

        # Generate dictionaries for inner and outer IDs
        to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = self.genFourDics(combinedDf)

        # Reindex combined DataFrame
        reIndexed_Combined = self.convertToInner(combinedDf, to_inner_uid, to_inner_iid)

        # Split the reindexed combined DataFrame back into train, val, and test sets
        train_len = len(self.train_df)
        val_len = len(self.val_df)

        self.train_df = reIndexed_Combined.iloc[:train_len, :]
        self.val_df = reIndexed_Combined.iloc[train_len:train_len + val_len, :]
        self.test_df = reIndexed_Combined.iloc[train_len + val_len:, :]

    def prepare_dataloader(self, df):
        if df is not None:
            # Convert user IDs, item IDs, and ratings to tensors, adjusting indices to start from 0.
            user_ids = torch.tensor(df.userID.values - 1, dtype=torch.long)
            item_ids = torch.tensor(df.itemID.values - 1, dtype=torch.long)
            ratings = torch.tensor(df.rating.values, dtype=torch.float32)

            # Create a TensorDataset
            data = TensorDataset(user_ids, item_ids, ratings)

            # Create a DataLoader for batch processing
            dataloader = DataLoader(data, batch_size=self.batch_size)

            return dataloader
        else:
            return None

