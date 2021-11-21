import os
import random
from .base import BaseDataset
import pickle
import h5py
import torch
import numpy as np
import json

class InputFeatures(object):
    """A single set of features of data."""    
    def __init__(self,
                 image_feat=None,
                 image_label=None,
                 image_mask=None,
                 pose_feat=None,
                 pose_loc=None,
                 pose_label=None,
                 pose_mask=None,
                 audio_feat = None,
                 audio_label = None, 
                 audio_target_label = None):
        self.image_feat = image_feat
        self.image_label = image_label
        self.image_mask = image_mask
        self.pose_feat = pose_feat
        self.pose_loc = pose_loc      
        self.pose_label = pose_label
        self.pose_mask = pose_mask 
        self.audio_feat = audio_feat
        self.audio_label = audio_label
        self.audio_target_label = audio_target_label 

class MUSICMixMultimodalDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixMultimodalDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.num_seq = 2
        self.root_path = "/ubc/cs/research/shield/datasets/MUSIC21_dataset/preprocessed_data/music21_bert_features/music_dataset"
        self.frame_root = "/ubc/cs/research/shield/datasets/MUSIC21_dataset/preprocessed_data/music21_all_frames"
        self.audio_root = "/ubc/cs/research/shield/datasets/MUSIC21_dataset/preprocessed_data/music21_bert_features/audio_feat"
        self.pose_root = "/ubc/cs/research/shield/datasets/MUSIC21_dataset/preprocessed_data/music21_bert_features/pose_feat_split_frame"
        self.pose_bbox_path = "/ubc/cs/research/shield/datasets/MUSIC21_dataset/preprocessed_data/music21_bert_features/alphapose_bbox_json"

        #audio classification  
        gt_lable_dict = {'bagpipe': 1, 'clarinet': 2, 'flute': 3, 'drum': 4, 'acoustic_guitar':5,'ukulele': 6, 'accordion': 7, 'bassoon': 8, 'guzheng': 9, 'xylophone': 10, 'erhu': 11, 'tuba':12, 'congas': 13, 'saxophone': 14, 'cello': 15, 'violin': 16, 'electric_bass': 17, 'piano':18,'banjo': 19, 'trumpet': 20, 'pipa': 21}
        solo_file_path = os.path.join(self.root_path, "MUSIC21_solo_videos.json")
        duet_file_path = os.path.join(self.root_path,"MUSIC_duet_videos.json") 
        dataDict_solo = None 
        dataDict_duet = None

        with open(solo_file_path,'r') as load_f:
            dataDict_solo = json.load(load_f)
        with open(duet_file_path,'r') as load_f:
            dataDict_duet = json.load(load_f)

        self.all_vid_label = {}

        for key, value in dataDict_solo["videos"].items(): 
            for item in value:
                self.all_vid_label[item] = [gt_lable_dict[key]] 

        for key, value in dataDict_duet["videos"].items(): 
            for item in value:
                self.all_vid_label[item] = [gt_lable_dict[key.split(' ')[0]], gt_lable_dict[key.split(' ')[1]]]


    def random_region(self,image_feat, num_boxes): 
        output_label = np.zeros([self.num_frames, num_boxes])
        for w in range(self.num_frames):
            for i in range(num_boxes):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15 :
                    prob /= 0.15 
                    # 80% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[w,i] = 0
                    # -> rest 10% randomly keep current token
                    output_label[w,i] = 1
                else:
                    output_label[w,i] = -1
        return image_feat, output_label

    def convert_example_to_features(self, image_feat, pose_feat, pose_loc, num_seq, audio_feat, label, target_label, num_frames):
        N = self.num_mix
        num_boxes = num_seq
        image_mask = []
        pose_mask = [] 
        image_label = [[] for n in range(N)]
        pose_label = [[] for n in range(N)] 
        for n in range(N): 
            image_feat[n], image_label[n] = self.random_region(image_feat[n], num_boxes)
            pose_feat[n], pose_label[n] = self.random_region(pose_feat[n], num_boxes)
            image_mask_tmp = [[1] * (num_boxes)] * self.num_frames 
            pose_mask_tmp = [[1] * (num_boxes)] * self.num_frames
  
            #Zero-pad up to the visual sequence length.
            while len(image_mask_tmp) < self.num_frames:
                image_mask_tmp.append(0)
                image_label[n].append(-1)
                pose_mask_tmp.append(0) 
                pose_label[n].append(-1)

            assert len(image_mask_tmp) == self.num_frames
            assert len(pose_mask_tmp) == self.num_frames
            image_mask.append(image_mask_tmp)
            pose_mask.append(pose_mask_tmp) 

        features = InputFeatures(
                   image_feat=image_feat,
                   image_label=np.array(image_label),
                   image_mask = np.array(image_mask),
                   pose_feat = pose_feat,
                   pose_loc = pose_loc,
                   pose_label = np.array(pose_label),
                   pose_mask = np.array(pose_mask), 
                   audio_feat = audio_feat, 
                   audio_label = label, 
                   audio_target_label = target_label)

        return features    

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        pose_features = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        pose_location = np.zeros((N, self.num_frames, self.num_seq, 5), dtype=np.float32)   
        final_pose_location = [None for n in range(N)]
        original_img_size = [None for n in range(N)]
        target_label = [[] for n in range(N)]  
        label = torch.zeros(N, 21) 

        # the first video
        index = self.check_video_frames_exists(self.frame_root, index)
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            indexN = self.check_video_frames_exists(self.frame_root, indexN)
            infos[n] = self.list_sample[indexN]

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)

        for n, infoN in enumerate(infos):
            #audio label for classification
            target = self.all_vid_label[infoN]  
            for j in range(len(target)): 
                label[n][target[j]-1]=1 
            #process for batch compatible
            if len(target) < self.num_seq: 
                target.append(-1)   
            target_label[n] = target 

            #load pose feat (generated by GCN)
            pose_feat_path = os.path.join(self.pose_root, infoN)
            pose_json = os.path.join(self.pose_bbox_path, infoN+".json") 
            with open(pose_json, "rb") as f: 
                pose_data = json.load(f) 

            path_frameN = os.path.join(self.frame_root,infoN) 
            path_audioN = os.path.join(self.audio_root,infoN+".wav") 
            count_framesN = len(os.listdir(pose_feat_path))  

            if self.split == 'train':
                # random, not to sample start and end n-frames
                if idx_margin+1 < int(count_framesN)-idx_margin:
                    center_frameN = random.randint(
                        idx_margin+1, int(count_framesN)-idx_margin)
                else:
                    center_frameN = int(count_framesN) // 2
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN
            
            vid_pose_feat = torch.zeros(self.num_frames, 2, 256, 68)
            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = i  
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset + 1))) 

                #load pose features
                try:
                    vid_pose_feat[i] = torch.from_numpy(np.load(os.path.join(pose_feat_path, '{:06d}.npy'.format(center_frameN + idx_offset)), allow_pickle=True))
                except Exception as e:
                    print("error in "+infoN)
                

                #load pose location (pose bbox)
                if '{:06d}.jpg'.format(center_frameN + idx_offset) in pose_data:
                    bbox_list = pose_data['{:06d}.jpg'.format(center_frameN + idx_offset)]
                    bbox_list.sort(key = lambda x: x[4], reverse=True)
                    for s, box in enumerate(bbox_list): 
                        if s < self.num_seq:
                            pose_location[n, i, s,:4] = box[:4] 
            pose_features[n] = vid_pose_feat
            path_audios[n] = path_audioN
           
        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n], original_img_size[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        self.image_size = 224  
        for n, infoN in enumerate(infos):
            for i in range(self.num_frames): 
                #rescale bbox which we get from alphapose
                if not original_img_size[n] is None: 
                    x_ = original_img_size[n][i][0]
                    y_ = original_img_size[n][i][1]
                    x_scale = self.image_size/x_
                    y_scale = self.image_size/y_
                else:
                    x_scale = 1
                    y_scale = 1
                for s in range(self.num_seq): 
                    pose_location[n,i,s,0] = pose_location[n,i,s,0] * x_scale
                    pose_location[n,i,s,1] = pose_location[n,i,s,1] * y_scale
                    pose_location[n,i,s,2] = pose_location[n,i,s,2] * x_scale
                    pose_location[n,i,s,3] = pose_location[n,i,s,3] * y_scale 
                    pose_location[n,i,s,2] = pose_location[n,i,s,2] + pose_location[n,i,s,0]
                    pose_location[n,i,s,3] = pose_location[n,i,s,3] + pose_location[n,i,s,1]

            #process pose location as vilbert  
            pose_location[n,i,:,4] = (pose_location[n,i,:,3] - pose_location[n,i,:,1]) * (pose_location[n,i,:,2] - pose_location[n,i,:,0]) / (float(self.image_size) * float(self.image_size))  
            pose_location[n,i,:,0] = pose_location[n,i,:,0] / float(self.image_size)  
            pose_location[n,i,:,1] = pose_location[n,i,:,1] / float(self.image_size) 
            pose_location[n,i,:,2] = pose_location[n,i,:,2] / float(self.image_size) 
            pose_location[n,i,:,3] = pose_location[n,i,:,3] / float(self.image_size) 

            final_pose_location[n] = pose_location[n]

        ###start of bert dataloader
        cur_features = self.convert_example_to_features(frames, pose_features, final_pose_location, self.num_seq, mag_mix, label, target_label, self.num_frames) 
        cur_tensors = (cur_features.image_feat,
                      cur_features.image_label,  
                      cur_features.image_mask,
                      cur_features.pose_feat,
                      cur_features.pose_loc,
                      cur_features.pose_label, 
                      cur_features.pose_mask,  
                      cur_features.audio_feat,
                      cur_features.audio_label, 
                      cur_features.audio_target_label)
        image_feat, image_label,image_mask, pose_feat, pose_loc,pose_label, pose_mask, audio_feat, audio_label, audio_target_label = cur_tensors
        
        image_feat_final = [None for n in range(N)] #torch.zeros((N,self.num_frames+1, 3, self.image_size,self.image_size))   
        image_mask_final = [None for n in range(N)] #torch.zeros((N,self.num_frames+1, self.num_seq))
        pose_feat_final = [None for n in range(N)] #torch.zeros((N, self.num_frames+1, self.num_seq, 256, 68))
        pose_mask_final = [None for n in range(N)] #torch.zeros((N,self.num_frames+1, self.num_seq))
        pose_loc_final = [None for n in range(N)] #torch.zeros((N,self.num_frames+1, self.num_seq,5))

        for n in range(N):
            #batch_size = image_feat[n].shape[0] 
            image_mask_tmp = image_mask[n] 
            image_feat_tmp = image_feat[n]
            if len(image_mask_tmp.shape) < 2:
                image_mask_tmp = image_mask_tmp.reshape(1,image_mask_tmp.shape[0])
            g_image_feat = np.sum(image_feat_tmp.numpy(), axis=1) / np.sum(image_mask_tmp) #, axis=1, keepdims=True)
            image_feat_tmp = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat_tmp.numpy()], axis=1)  
            image_feat_tmp = np.array(image_feat_tmp, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1, 1]]), 1, axis=0) 
            image_mask_tmp = np.concatenate([g_image_mask, image_mask_tmp], axis=0) 
            image_feat_final[n]= torch.from_numpy(image_feat_tmp) 
            image_mask_final[n] = torch.from_numpy(image_mask_tmp)  
            
            pose_mask_tmp = pose_mask[n] 
            pose_feat_tmp = pose_feat[n]
            pose_loc_tmp = pose_loc[n] 
            if len(pose_mask_tmp.shape) < 2:  
                pose_mask_tmp = pose_mask_tmp.reshape(1,pose_mask_tmp.shape[0])
            g_pose_feat = np.sum(pose_feat_tmp.numpy(), axis=0) / np.sum(pose_mask_tmp) #, axis=1, keepdims=True)
            pose_feat_tmp = np.concatenate([np.expand_dims(g_pose_feat, axis=0), pose_feat_tmp.numpy()], axis=0)
            pose_feat_tmp = np.array(pose_feat_tmp, dtype=np.float32) 
            g_pose_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), 2, axis=0)
            pose_loc_tmp = np.concatenate([np.expand_dims(g_pose_loc, axis=0), pose_loc_tmp], axis=0)
            pose_loc_tmp = np.array(pose_loc_tmp, dtype=np.float32) 
            g_pose_mask = np.repeat(np.array([[1,1]]), 1, axis=0)   
            pose_mask_tmp = np.concatenate([g_pose_mask, pose_mask_tmp], axis=0)   
            pose_feat_final[n] = torch.from_numpy(pose_feat_tmp) 
            pose_mask_final[n] = torch.from_numpy(pose_mask_tmp)
            pose_loc_final[n] = torch.from_numpy(pose_loc_tmp)

        ###end of bert dataloader
        
        #ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'pose_feat': pose_features, 'pose_loc': final_pose_location, 'label': label, 'target_label': target_label}

        ret_dict = {'mag_mix': audio_feat, 'frames': image_feat_final, 'image_label': image_label, 'image_mask': image_mask_final, 'mags': mags, 'pose_feat': pose_feat_final, 'pose_loc': pose_loc_final, 'pose_mask': pose_mask_final,'pose_label':pose_label, 'label': label, 'target_label': target_label}

        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
