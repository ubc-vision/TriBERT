import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import tribert_retrieval_networks
import orig_retrieval_networks
from retrieval_datasets import *
from tqdm import tqdm


def train(model, criterion, optimizer, dataloader, device, args):
    CKPT_DIR = f'./checkpoints/orig/{args.exp_name}'
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    loss_str = []
    best_loss = 1000000
    for i in range(args.epochs):
        accum_loss = 0
        batch_cnt = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Map all tensors to device
            batch[-1] = batch[-1].to(device)
            for data_dict in batch[:-1]:
                for key, value in data_dict.items():
                    data_dict[key] = data_dict[key].to(device)

            if args.retrieval_mode == 'visaud2pose':
                pos, neg1, neg2, hard_neg, target = batch[0], batch[1], batch[2], batch[3], batch[4]
                scores = model(pos, neg1, neg2, hard_neg)  # scores is of shape (batch_size, 3)

            else:
                pos, neg1, neg2, target = batch[0], batch[1], batch[2], batch[3]
                scores = model(pos, neg1, neg2)  # scores is of shape (batch_size, 3)

            loss = criterion(scores, target)

            loss_str.append(loss.item())
            accum_loss += loss.item()
            batch_cnt += 1

            loss.backward()
            optimizer.step()

        avg_loss = accum_loss / batch_cnt

        if avg_loss < best_loss:
            best_loss = avg_loss
            CKPT_PATH = os.path.join(CKPT_DIR, f'{args.exp_name}.pt')
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, CKPT_PATH)

    loss_arr = np.asarray(loss_str)
    smallest_loss = loss_arr.min()
    np.save(os.path.join(CKPT_DIR, f'{args.exp_name}_loss_{smallest_loss:.2f}.npy'), loss_arr)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--exp_name', type=str, help='name of experiment. Will be used to name saved checkpoints')
    parser.add_argument('--embeddings_path', type=str, help='path of folder containing the .npy embedding files')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)

    # Model settings
    parser.add_argument('--retrieval_mode', type=str, choices=['aud2vis', 'vis2aud', 'aud2pose', 'pose2aud', 'visaud2pose'])
    parser.add_argument('--embedding', type=str, choices=['orig', 'tribert'],
                        help='retrieval using baseline representations or tribert representations')
    args = parser.parse_args()

    '''
    Paths to vision, audio, and pose embeddings, saved as a numpy array in a .npy file. Each of these are of shape 
    (n, k), where n is the number of embeddings and k the dimensionality of each embedding.
    
    NOTE: Be sure to use the correct type of embeddings (either baseline or tribert)
    '''
    VIS_EMBED_PATH = os.path.join(args.embeddings_path, 'train_vision.npy')
    AUD_EMBED_PATH = os.path.join(args.embeddings_path, 'train_audio.npy')
    POS_EMBED_PATH = os.path.join(args.embeddings_path, 'train_pose.npy')

    device = 'cuda'

    np.random.seed(1)
    torch.manual_seed(1)

    # Load dataset and loader
    if args.retrieval_mode == 'visaud2pose':
        train_dataset = MultiModalRetrievalDataset(vision_embed=VIS_EMBED_PATH,
                                                   audio_embed=AUD_EMBED_PATH,
                                                   pose_embed=POS_EMBED_PATH)
    else:

        train_dataset = SingleModalRetrievalDataset(vision_embed=VIS_EMBED_PATH,
                                                    audio_embed=AUD_EMBED_PATH,
                                                    pose_embed=POS_EMBED_PATH)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    if args.embedding == 'tribert':
        if args.retrieval_mode == 'aud2vis':
            model = tribert_retrieval_networks.aud2vis()
        elif args.retrieval_mode == 'vis2aud':
            model = tribert_retrieval_networks.vis2aud()
        elif args.retrieval_mode == 'aud2pose':
            model = tribert_retrieval_networks.aud2pose()
        elif args.retrieval_mode == 'pose2aud':
            model = tribert_retrieval_networks.pose2aud()
        else:
            model = tribert_retrieval_networks.visaud2pose()

    else:
        if args.retrieval_mode == 'aud2vis':
            model = orig_retrieval_networks.aud2vis()
        elif args.retrieval_mode == 'vis2aud':
            model = orig_retrieval_networks.vis2aud()
        elif args.retrieval_mode == 'aud2pose':
            model = orig_retrieval_networks.aud2pose()
        elif args.retrieval_mode == 'pose2aud':
            model = orig_retrieval_networks.pose2aud()
        else:
            model = orig_retrieval_networks.visaud2pose()

    model = model.to(device)

    # Train
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    model = train(model, criterion, optimizer, dataloader, device, args)
