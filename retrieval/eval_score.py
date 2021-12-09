import argparse
import torch
import torch.nn.functional as F

import numpy as np
import tribert_retrieval_networks
import orig_retrieval_networks
from tqdm import tqdm


def evaluate(model, test_vision, test_audio, test_pose, rand_idx, args, device):
    
    num_eval = test_vision.shape[0]
    
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    
    # Get query embeddings and target embedding 
    for i in tqdm(range(num_eval)):
        if args.retrieval_variant == 'aud2vis':
            query = torch.from_numpy(test_audio[rand_idx[i], ...]).float().to(device)
            result = torch.from_numpy(test_vision[rand_idx[i], ...]).float().to(device)
        if args.retrieval_variant == 'vis2aud':
            query = torch.from_numpy(test_vision[rand_idx[i], ...]).float().to(device)
            result = torch.from_numpy(test_audio[rand_idx[i], ...]).float().to(device)
        if args.retrieval_variant == 'aud2pose':
            query = torch.from_numpy(test_audio[rand_idx[i], ...]).float().to(device)
            result = torch.from_numpy(test_pose[rand_idx[i], ...]).float().to(device)
        if args.retrieval_variant == 'pose2aud':
            query = torch.from_numpy(test_pose[rand_idx[i], ...]).float().to(device)
            result = torch.from_numpy(test_audio[rand_idx[i], ...]).float().to(device)
        if args.retrieval_variant == 'vis+aud2pose':
            vision = torch.from_numpy(test_vision[rand_idx[i], ...]).float().to(device)
            audio = torch.from_numpy(test_audio[rand_idx[i], ...]).float().to(device)
            query = model.fuse_forward(vision, audio)
            result = torch.from_numpy(test_pose[rand_idx[i], ...]).float().to(device)

        scores = torch.zeros(num_eval).to(device)

        for j in range(test_vision.shape[0]):
            score = model.forward_once(query, result)
            scores[j] = score

        scores = F.softmax(scores, dim=0)
        ordered_scores = torch.argsort(scores, descending=True)

        if rand_idx[i] in ordered_scores[:1]:
            correct_1 += 1

        if rand_idx[i] in ordered_scores[:5]:
            correct_5 += 1

        if rand_idx[i] in ordered_scores[:10]:
            correct_10 += 1

    acc_1 = correct_1 / num_eval
    acc_5 = correct_5 / num_eval
    acc_10 = correct_10 / num_eval

    return acc_1, acc_5, acc_10


def test(tribert_model, orig_model, tribert_embeddings, orig_embeddings, device, args):

    # Unpack embeddings
    tribert_vis = tribert_embeddings['vision']
    tribert_aud = tribert_embeddings['audio']
    tribert_pose = tribert_embeddings['pose']

    orig_vis = orig_embeddings['vision']
    orig_aud = orig_embeddings['audio']
    orig_pose = orig_embeddings['pose']
    
    num_tribert = tribert_vis.shape[0]
    num_orig = orig_vis.shape[0]

    print(f'Evaluating on {num_tribert} tribert embeddings and {num_orig} original embeddings...')

    # Shuffle the embeddings
    rand_tribert = np.arange(num_tribert)
    rand_orig = np.arange(num_orig)
    np.random.shuffle(rand_tribert)
    np.random.shuffle(rand_orig)
    
    # Evaluate retrieval on tribert embeddings 
    tribert_acc1, tribert_acc5, tribert_acc10 = evaluate(tribert_model, tribert_vis, tribert_aud, tribert_pose, rand_tribert, args, device)

    # Evaluate retrieval on original embeddings 
    orig_acc1, orig_acc5, orig_acc10 = evaluate(orig_model, orig_vis, orig_aud, orig_pose, rand_orig, args, device)
    
    return (tribert_acc1, tribert_acc5, tribert_acc10), (orig_acc1, orig_acc5, orig_acc10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tribert_embedding_path', type=str, default='retrieval_embeddings/tribert_test.pt')
    parser.add_argument('--orig_embedding_path', type=str, default='retrieval_embeddings/orig_test.pt')
    parser.add_argument('--tribert_path', type=str, help='path of tribert embeddings retrieval network checkpoint')
    parser.add_argument('--orig_path', type=str, help='path of original embeddings retrieval network checkpoint')
    parser.add_argument('--retrieval_variant', type=str, choices=['aud2vis', 'vis2aud', 'aud2pose', 'pose2aud', 'vis+aud2pose'], help='select 1 of the 5 retrieval variants shown in the paper')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    '''
    tribert_embeddings and orig_embeddings are dictionaries saved as a pytorch .pt file. Each .pt file has 3 keys: 
    vision, audio, and pose. The value for each key is the corresponding modality embedding, which has shape (n, k), 
    where n is the number of embeddings and k is the dimensionality of the embedding
    '''
    tribert_embeddings = torch.load(args.tribert_embedding_path)
    orig_embeddings = torch.load(args.orig_embedding_path)

    np.random.seed(10)
    torch.manual_seed(1)

    tribert_ckpt = torch.load(args.tribert_path, map_location=device)
    orig_ckpt = torch.load(args.orig_path, map_location=device)

    # Load retrieval model checkpoints
    if args.retrieval_variant == 'aud2vis':
        tribert = tribert_ckpt['aud2vis']
        orig = orig_ckpt['aud2vis']
        tribert_model = tribert_retrieval_networks.aud2vis()
        orig_model = orig_retrieval_networks.aud2vis()
    if args.retrieval_variant == 'vis2aud':
        tribert = tribert_ckpt['vis2aud']
        orig = orig_ckpt['vis2aud']
        tribert_model = tribert_retrieval_networks.vis2aud()
        orig_model = orig_retrieval_networks.vis2aud()
    if args.retrieval_variant == 'aud2pose':
        tribert = tribert_ckpt['aud2pose']
        orig = orig_ckpt['aud2pose']
        tribert_model = tribert_retrieval_networks.aud2pose()
        orig_model = orig_retrieval_networks.aud2pose()
    if args.retrieval_variant == 'pose2aud':
        tribert = tribert_ckpt['pose2aud']
        orig = orig_ckpt['pose2aud']
        tribert_model = tribert_retrieval_networks.pose2aud()
        orig_model = orig_retrieval_networks.pose2aud()
    if args.retrieval_variant == 'vis+aud2pose':
        tribert = tribert_ckpt['vis+aud2pose']
        orig = orig_ckpt['vis+aud2pose']
        tribert_model = tribert_retrieval_networks.visaud2pose()
        orig_model = orig_retrieval_networks.visaud2pose()

    tribert_model, orig_model = tribert_model.to(device), orig_model.to(device)
    tribert_model.load_state_dict(tribert['model_state_dict'])
    orig_model.load_state_dict(orig['model_state_dict'])

    tribert_model.eval()
    orig_model.eval()

    # Evaluate retrieval
    tribert_acc, orig_acc = test(tribert_model, orig_model, tribert_embeddings, orig_embeddings, device, args)

    print("*" * 80)
    print("TriBERT embedding retrieval results\n")
    print("*" * 80)
    print(f'Top-1 Acc: {tribert_acc[0] * 100:.2f}%, Top-5 Acc: {tribert_acc[1] * 100:.2f}%, Top-10 Acc: {tribert_acc[2] * 100:.2f}%')

    print("*" * 80)
    print("Original embedding retrieval results\n")
    print("*" * 80)
    print(f'Top-1 Acc: {orig_acc[0] * 100:.2f}%, Top-5 Acc: {orig_acc[1] * 100:.2f}%, Top-10 Acc: {orig_acc[2] * 100:.2f}%')
