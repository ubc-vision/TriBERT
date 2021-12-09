import torch
import torch.nn as nn
import torch.nn.functional as F


class aud2vis(nn.Module):

    def __init__(self):
        super(aud2vis, self).__init__()

        # MLP to map audio embedding from (4096) -> (8192) dim tensor
        self.audio_transform = nn.Sequential(nn.Linear(4096, 8192))

        # MLP to compute alignment score
        self.align_net = nn.Sequential(nn.Linear(8192, 4096),
                                       nn.Tanh(),
                                       nn.Linear(4096, 2048),
                                       nn.Tanh(),
                                       nn.Linear(2048, 1024),
                                       nn.Tanh(),
                                       nn.Linear(1024, 1))

    # Single forward pass with one sample pair
    def forward_once(self, audio, vision):
        # audio is now 8192-dimensional
        audio = self.audio_transform(audio)

        # Calculate alignment score
        align_score = self.align_net(audio * vision)
        return align_score

    def forward(self, sample, neg1, neg2):
        audio_anchor = sample['audio']

        # Forward pass for positive pair
        pos_score = self.forward_once(audio_anchor, sample['vision'])

        # Forward pass for negative pair 1
        neg1_score = self.forward_once(audio_anchor, neg1['vision'])

        # Forward pass for negative pair 2
        neg2_score = self.forward_once(audio_anchor, neg2['vision'])

        return F.softmax(torch.cat((pos_score, neg1_score, neg2_score), dim=1), dim=1)


class vis2aud(nn.Module):

    def __init__(self):
        super(vis2aud, self).__init__()

        # MLP to map audio embedding from (4096) -> (8192) dim tensor
        self.audio_transform = nn.Sequential(nn.Linear(4096, 8192))

        # MLP to compute alignment score
        self.align_net = nn.Sequential(nn.Linear(8192, 4096),
                                       nn.Tanh(),
                                       nn.Linear(4096, 2048),
                                       nn.Tanh(),
                                       nn.Linear(2048, 1024),
                                       nn.Tanh(),
                                       nn.Linear(1024, 1))

    # Single forward pass with one sample pair
    def forward_once(self, vision, audio):
        # audio is now 8192-dimensional
        audio = self.audio_transform(audio)

        # Calculate alignment score
        align_score = self.align_net(audio * vision)
        return align_score

    def forward(self, sample, neg1, neg2):
        vision_anchor = sample['vision']

        # Forward pass for positive pair
        pos_score = self.forward_once(vision_anchor, sample['audio'])

        # Forward pass for negative pair 1
        neg1_score = self.forward_once(vision_anchor, neg1['audio'])

        # Forward pass for negative pair 2
        neg2_score = self.forward_once(vision_anchor, neg2['audio'])

        return F.softmax(torch.cat((pos_score, neg1_score, neg2_score), dim=1), dim=1)


class aud2pose(nn.Module):

    def __init__(self):
        super(aud2pose, self).__init__()

        # MLP to map audio embedding from (4096) -> (8192) dim tensor
        self.audio_transform = nn.Sequential(nn.Linear(4096, 8192))

        # MLP to compute alignment score
        self.align_net = nn.Sequential(nn.Linear(8192, 4096),
                                       nn.Tanh(),
                                       nn.Linear(4096, 2048),
                                       nn.Tanh(),
                                       nn.Linear(2048, 1024),
                                       nn.Tanh(),
                                       nn.Linear(1024, 1))

    # Single forward pass with one sample pair
    def forward_once(self, audio, pose):
        # audio is now 8192-dimensional
        audio = self.audio_transform(audio)

        # Calculate alignment score
        align_score = self.align_net(audio * pose)
        return align_score

    def forward(self, sample, neg1, neg2):
        audio_anchor = sample['audio']

        # Forward pass for positive pair
        pos_score = self.forward_once(audio_anchor, sample['pose'])

        # Forward pass for negative pair 1
        neg1_score = self.forward_once(audio_anchor, neg1['pose'])

        # Forward pass for negative pair 2
        neg2_score = self.forward_once(audio_anchor, neg2['pose'])

        return F.softmax(torch.cat((pos_score, neg1_score, neg2_score), dim=1), dim=1)


class pose2aud(nn.Module):

    def __init__(self):
        super(pose2aud, self).__init__()

        # MLP to map audio embedding from (4096) -> (8192) dim tensor
        self.audio_transform = nn.Sequential(nn.Linear(4096, 8192))

        # MLP to compute alignment score
        self.align_net = nn.Sequential(nn.Linear(8192, 4096),
                                       nn.Tanh(),
                                       nn.Linear(4096, 2048),
                                       nn.Tanh(),
                                       nn.Linear(2048, 1024),
                                       nn.Tanh(),
                                       nn.Linear(1024, 1))

    # Single forward pass with one sample pair
    def forward_once(self, pose, audio):
        # audio is now 8192-dimensional
        audio = self.audio_transform(audio)

        # Calculate alignment score
        align_score = self.align_net(audio * pose)
        return align_score

    def forward(self, sample, neg1, neg2):
        pose_anchor = sample['pose']

        # Forward pass for positive pair
        pos_score = self.forward_once(pose_anchor, sample['audio'])

        # Forward pass for negative pair 1
        neg1_score = self.forward_once(pose_anchor, neg1['audio'])

        # Forward pass for negative pair 2
        neg2_score = self.forward_once(pose_anchor, neg2['audio'])

        return F.softmax(torch.cat((pos_score, neg1_score, neg2_score), dim=1), dim=1)


class visaud2pose(nn.Module):

    def __init__(self):
        super(visaud2pose, self).__init__()

        # MLP to map audio embedding from (4096) -> (8192) dim tensor
        self.audio_transform = nn.Sequential(nn.Linear(4096, 8192))

        # MLP for fusing audio and pose embeddings
        self.fuse = nn.Sequential(nn.Linear(8192, 8192),
                                  nn.Tanh(),
                                  nn.Linear(8192, 8192),
                                  nn.Tanh(),
                                  nn.Linear(8192, 8192)
                                  )

        # MLP to compute alignment score
        self.align_net = nn.Sequential(nn.Linear(8192, 4096),
                                       nn.Tanh(),
                                       nn.Linear(4096, 2048),
                                       nn.Tanh(),
                                       nn.Linear(2048, 1024),
                                       nn.Tanh(),
                                       nn.Linear(1024, 1))

    def fuse_forward(self, vision, audio):
        # audio is now 8192-dimensional
        audio = self.audio_transform(audio)

        audio = audio.view(-1, 4, 2, 1024)
        vision = vision.view(-1, 4, 2, 1024)

        # Fuse audio and pose embeddings
        fuse_inter = (F.softmax(audio * vision, dim=-1) * vision) + audio
        fuse_inter = fuse_inter.view(-1, 4*2*1024)
        fuse_embed = self.fuse(fuse_inter) + fuse_inter


        return fuse_embed

    # Single forward pass with one sample pair
    def forward_once(self, fuse, pose):

        # Calculate alignment score
        align_score = self.align_net(fuse * pose)
        return align_score

    def forward(self, sample, neg1, neg2, hard_neg):

        fuse_anchor = self.fuse_forward(sample['vision'], sample['audio'])

        # Forward pass for positive pair
        pos_score = self.forward_once(fuse_anchor, sample['pose'])

        # Forward pass for negative pair 1
        neg1_score = self.forward_once(fuse_anchor, neg1['pose'])

        # Forward pass for negative pair 2
        neg2_score = self.forward_once(fuse_anchor, neg2['pose'])

        hardneg_score = self.forward_once(fuse_anchor, hard_neg['pose'])

        return F.softmax(torch.cat((pos_score, neg1_score, neg2_score, hardneg_score), dim=1), dim=1)