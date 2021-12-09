from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import Dataset
import numpy as np

'''
Dataset used for training single modality -> single modality retrieval (eg. aud2vis, pose2aud, etc.). These retrieval 
networks are trained using 3-way multiple choice, with one positive pair and 2 hard negatives. Hard negatives are 
sampled using nearest neighbours.
'''
class SingleModalRetrievalDataset(Dataset):

    def __init__(self, vision_embed, audio_embed, pose_embed, vision_trunc=None):
        self.vision_embed = np.load(vision_embed)
        self.audio_embed = np.load(audio_embed)
        self.pose_embed = np.load(pose_embed)

        # Run Nearest Neighbours for selecting negatives
        self.neigh = NearestNeighbors(n_neighbors=25)
        self.neigh.fit(self.vision_embed)

    def __len__(self):
        return self.vision_embed.shape[0]

    def __getitem__(self, idx):
        # Load sample embeddings
        vision = self.vision_embed[idx, ...]  # (3*4*224*224)
        audio = self.audio_embed[idx, ...]  #
        pose = self.pose_embed[idx, ...]  #

        # Create 3 negative pairs of audio/pose by uniformly sampling from current image's k nearest neighbors
        '''
        neg pair 1: correct audio, wrong pose
        neg pair 2: wrong audio, correct pose 
        neg pair 3: wrong audio, wrong pose
        '''
        neigh_ind = self.neigh.kneighbors([vision], return_distance=False).squeeze()
        neg_ind = neigh_ind[np.random.randint(low=1, high=25, size=3)]

        sample = {'vision': torch.from_numpy(vision).float(),
                  'audio': torch.from_numpy(audio).float(),
                  'pose': torch.from_numpy(pose).float()}

        neg1 = {'audio': torch.from_numpy(audio).float(),
                'pose': torch.from_numpy(self.pose_embed[neg_ind[0], ...]).float()}

        neg2 = {'audio': torch.from_numpy(self.audio_embed[neg_ind[1], ...]).float(),
                'pose': torch.from_numpy(pose).float()}


        target = 0  # In the output of model(), the first score is for the positive pair

        return sample, neg1, neg2, neg3, target


'''
Used to train visaud2pose. 4-way multiple choice with one positive pair, two easy negatives, and a hard negative. 
Easy negatives are sampled randomly from the dataset while hard negatives are sampled using nearest neighbours.
'''
class MultiModalRetrievalDataset(Dataset):

    def __init__(self, vision_embed, audio_embed, pose_embed):
        self.vision_embed = np.load(vision_embed)
        self.audio_embed = np.load(audio_embed)
        self.pose_embed = np.load(pose_embed)

        # Run Nearest Neighbours for selecting negatives
        self.neigh = NearestNeighbors(n_neighbors=25)
        self.neigh.fit(self.vision_embed)

    def __len__(self):
        return self.vision_embed.shape[0]

    def __getitem__(self, idx):
        # Load sample embeddings
        vision = self.vision_embed[idx, ...]  # (8192)
        audio = self.audio_embed[idx, ...]  # (4096)
        pose = self.pose_embed[idx, ...]  # (8192)

        # Create 3 negative pairs of audio/pose by uniformly sampling from current image's k nearest neighbors
        '''
        neg pair 1: correct audio, wrong pose
        neg pair 2: wrong audio, correct pose 
        neg pair 3: wrong audio, wrong pose
        '''
        neigh_ind = self.neigh.kneighbors([vision], return_distance=False).squeeze()
        hard_neg_ind = neigh_ind[np.random.randint(low=3, high=25, size=1)]
        neg_ind = np.random.randint(self.vision_embed.shape[0], size=2)

        sample = {'vision': torch.from_numpy(vision).float(),
                  'audio': torch.from_numpy(audio).float(),
                  'pose': torch.from_numpy(pose).float()}

        # Easy negatives
        neg1 = {'vision': torch.from_numpy(self.vision_embed[neg_ind[0], ...]).float(),
                'audio': torch.from_numpy(self.audio_embed[neg_ind[0], ...]).float(),
                'pose': torch.from_numpy(self.pose_embed[neg_ind[0], ...]).float()}

        neg2 = {'vision': torch.from_numpy(self.vision_embed[neg_ind[1], ...]).float(),
                'audio': torch.from_numpy(self.audio_embed[neg_ind[1], ...]).float(),
                'pose': torch.from_numpy(self.pose_embed[neg_ind[1], ...]).float()}

        # Hard negative
        neg3 = {'vision': torch.from_numpy(self.vision_embed[hard_neg_ind[0], ...]).float(),
                'audio': torch.from_numpy(self.audio_embed[hard_neg_ind[0], ...]).float(),
                'pose': torch.from_numpy(self.pose_embed[hard_neg_ind[0], ...]).float()}

        target = 0  # In the output of model(), the first score is for the positive pair

        return sample, neg1, neg2, neg3, target
