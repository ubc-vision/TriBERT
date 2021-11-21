import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertPoseEmbeddings(nn.Module):
    """Construct the embeddings from pose, spatial location (omit now) and token_type embeddings."""
    def __init__(self, config):
        super(BertPoseEmbeddings, self).__init__()
        self.pose_embeddings = nn.Linear(256*68, config.p_hidden_size)
        self.pose_location_embeddings = nn.Linear(5, config.p_hidden_size)
        self.LayerNorm = BertLayerNorm(config.p_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        pose_inputs = input_ids.view(batch_size, seq_len, -1)
        pose_embeddings = self.pose_embeddings(pose_inputs)
        loc_embeddings = self.pose_location_embeddings(input_loc)
        embeddings = self.LayerNorm(pose_embeddings+loc_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class vggish(nn.Module):
    def __init__(self):  
        super(vggish, self).__init__()   
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) 
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)     
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)     
        self.conv5 = nn.Conv2d(256,512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)    
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.relu6 = nn.ReLU() 
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)     
        self.relu7 = nn.ReLU() 
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X): 
        X = self.pool1(self.relu1(self.conv1(X))) 
        X = self.pool2(self.relu2(self.conv2(X))) 
        X = self.pool3(self.relu3(self.conv3(X)))    
        X = self.pool4(self.relu4(self.conv4(X)))  
        X = self.pool5(self.relu5(self.conv5(X))) 
        X = self.pool6(self.relu6(self.conv6(X)))
        X = self.pool7(self.relu7(self.conv7(X))) 
        X = X.view(X.shape[0],-1)
        return X 

class BertAudioEmbeddings(nn.Module):   
    def __init__(self, config):
        super(BertAudioEmbeddings, self).__init__()
        self.vggish_feat = vggish() 
        self.audio_embeddings = nn.Linear(config.a_feature_size, config.a_hidden_size) 
        self.audio_location_embeddings = nn.Linear(5, config.a_hidden_size)
        self.LayerNorm = BertLayerNorm(config.a_hidden_size, eps=1e-12)   
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):    
        vggish_feat = self.vggish_feat(input_ids) 
        audio_embeddings = self.audio_embeddings(vggish_feat)
        embeddings = self.LayerNorm(audio_embeddings)   
        embeddings = self.dropout(embeddings) 
        return embeddings

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class Resnet(nn.Module):
    def __init__(self, original_resnet):
        super(Resnet, self).__init__()
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1])
        # for param in self.features.parameters():
        # 	param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1))
        return x


class ResnetFC(nn.Module):
    def __init__(self, original_resnet, fc_dim=64,
                 pool_type='maxpool',config=None, conv_size=3):
        super(ResnetFC, self).__init__()
        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            2048, 1024, kernel_size=conv_size, padding=conv_size//2)

        self.pose_embeddings = BertPoseEmbeddings(config)
        self.audio_embeddings = BertAudioEmbeddings(config)
        
        #CAM feature
        self.relu = nn.ReLU() 
        self.dp1 = nn.Dropout(p=0.2)
        self.conv1x1_1 = nn.Conv2d(1024, 21, kernel_size=1)
        self.conv1x1_11 = nn.Conv2d(21, 21, kernel_size=1)
        self.dp2 = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1_2 = nn.Conv2d(1024, 21, kernel_size=1)    

        #BERT model
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        #weakly supervised loss
        self.image_fc_c = nn.Linear(config.v_target_size, config.v_target_size)
        self.image_fc_d = nn.Linear(config.v_target_size, config.v_target_size) 

        self.pose_fc_c = nn.Linear(config.v_target_size, config.v_target_size)
        self.pose_fc_d = nn.Linear(config.v_target_size, config.v_target_size) 

    def forward(self, x, pose_feat, pool=True):
        pose_feat = self.pose_embeddings(pose_feat) 
        pose_feat = pose_feat.permute(0,2,1).contiguous()
        pose_feat = F.adaptive_max_pool1d(pose_feat,1)

        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        pose_feat = pose_feat.view(pose_feat.size(0), pose_feat.size(1)) 
        return x

    def forward_multiframe(self, x, pose_feat, pose_loc, audio_feat, cam_map, image_feat, image_mask, pose_mask, audio_mask, co_attention_mask, pool=True):
  
        (B, C, T, H, W) = x.size()    
        seq_len = pose_feat.shape[2]
        feat_size = pose_feat.shape[3]
        key_points = pose_feat.shape[4]
        
        pose_feat = pose_feat.view(B*T,seq_len, feat_size, key_points)
        pose_loc = pose_loc.view(B*T, seq_len, -1) 
        pose_feat = self.pose_embeddings(pose_feat, pose_loc)
        pose_feat = pose_feat.view(B, T, seq_len, -1)
       
        audio_feat = self.audio_embeddings(audio_feat)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()

        #weakly supervised visual feature using CAM
        seg_feat = x
        seg_feat_1 = self.dp2(self.conv1x1_1(self.dp1(seg_feat))) 
        seg_feat_1 = self.relu(seg_feat_1)    
        seg_feat_1 = self.softmax(self.relu(self.conv1x1_11(seg_feat_1)))
        seg_feat_2 = self.conv1x1_2(seg_feat)
        seg_feat = seg_feat_1 * seg_feat_2
        classifier = F.avg_pool2d(seg_feat, kernel_size=7).view(seg_feat.shape[0], -1)    
        
        x = x.view(B, T, C, H, W)
        seg_feat = seg_feat.view(B,T,-1,H,W)
        classifier = classifier.view(B,T,-1)

        for i in range(B):  
            seq_classifier = classifier[i]
            value, index = seq_classifier.sort(descending=True)
            for j in range(T):
                cam_map[i,j,0,:,:]=seg_feat[i,j,index[j][0],:,:]
                cam_map[i,j,1,:,:]=seg_feat[i,j,index[j][1],:,:] 
       
        cam_map_feat = cam_map.clone().unsqueeze(2) 
        for i in range(2):
            image_feat[:,:,i]  = cam_map_feat[:,:,:,i] * x
        
        image_feature = image_feat.clone().view(-1, C, H, W)
        image_feature = F.avg_pool2d(image_feature, kernel_size=7)
        image_feature = image_feature.view(B, T, -1, C)
         
        """BERT model with multi modal pre-training heads."""
        sequence_output_v, sequence_output_p, sequence_output_a, pooled_output_v, pooled_output_p, pooled_output_a, all_attention_mask = self.bert(image_feature, pose_feat, audio_feat, image_mask, pose_mask, audio_mask, co_attention_mask)

        prediction_scores_v, prediction_scores_p, prediction_scores_a, seq_relationship_score = self.cls(sequence_output_v, sequence_output_p, sequence_output_a, pooled_output_v, pooled_output_p, pooled_output_a)

        #weakly supervised loss
        prediction_scores_v = prediction_scores_v.mean(1)
        prediction_scores_p = prediction_scores_p.mean(1)  
        prediction_scores_a = prediction_scores_a.mean(1)
        seq_relationship_score = seq_relationship_score.mean(1)
        
        image_classification_scores = F.softmax(self.image_fc_c(prediction_scores_v), dim=2)  #batch_size, seq, class_scores
        image_detection_scores = F.softmax(self.image_fc_d(prediction_scores_v), dim=1)
        image_combined_scores = image_classification_scores * image_detection_scores
        image_level_scores = torch.sum(image_combined_scores, dim=1)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        
        pose_classification_scores = F.softmax(self.pose_fc_c(prediction_scores_p), dim=2)
        pose_detection_scores = F.softmax(self.pose_fc_d(prediction_scores_p), dim=1)
        pose_combined_scores = pose_classification_scores * pose_detection_scores
        pose_level_scores = torch.sum(pose_combined_scores, dim=1)
        pose_level_scores = torch.clamp(pose_level_scores, min=0.0, max=1.0)

        prediction_scores_a = torch.mean(prediction_scores_a, dim=1)
        audio_classification_scores = F.softmax(prediction_scores_a, dim=1)

        #return x
        return seq_relationship_score, image_level_scores, pose_level_scores, audio_classification_scores

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.posePredictions = BertImagePredictionHead(config)
        self.audioPredictions = BertAudioPredictionHead(config)
        self.fusion_method = 'cat' #config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, sequence_output_v, sequence_output_p, sequence_output_a, pooled_output_v, pooled_output_p, pooled_output_a):
        
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_p + pooled_output_v + pooled_output_a)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_p * pooled_output_v * pooled_output_a)
        elif self.fusion_method == 'cat':
            pooled_output = self.dropout(torch.cat((pooled_output_v, pooled_output_p, pooled_output_a), dim=2))
        else:
            assert False
        
        prediction_scores_v = self.imagePredictions(sequence_output_v)
        prediction_scores_p = self.posePredictions(sequence_output_p)
        prediction_scores_a = self.audioPredictions(sequence_output_a)

        return prediction_scores_v, prediction_scores_p, prediction_scores_a, pooled_output #seq_relationship_score

class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertAudioPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertAudioPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.a_hidden_size, config.a_hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.a_hidden_act
        self.LayerNorm = BertLayerNorm(config.a_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertAudioPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertAudioPredictionHead, self).__init__()
        self.transform = BertAudioPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.a_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer")."""
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.encoder = BertEncoder(config)
        self.p_pooler = BertPosePooler(config)
        self.v_pooler = BertImagePooler(config)
        self.a_pooler = BertAudioPooler(config)

    def forward(self, image_feat, pose_feat, audio_feat, image_mask, pose_mask, audio_mask, co_attention_mask, output_all_encoded_layers=False, output_all_attention_masks=False):
        
        if image_mask is None:
            image_mask = torch.ones(
                image_feat.size(0), image_feat.size(1), image_feat.size(2)
            ).type_as(image_feat)

        if pose_mask is None:
            pose_mask = torch.ones(
                pose_feat.size(0), pose_feat.size(1), pose_feat.size(2)
            ).type_as(pose_feat)
       
        if audio_mask is None:
            audio_mask = torch.ones(
                image_feat.size(0), image_feat.size(1), image_feat.size(2)
            ).type_as(audio_feat)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_pose_attention_mask = pose_mask.unsqueeze(1)
        extended_image_attention_mask = image_mask.unsqueeze(1)
        extended_audio_attention_mask = audio_mask.unsqueeze(1)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_pose_attention_mask = extended_pose_attention_mask.to(
            dtype=torch.float32 #next(self.parameters()).dtype
        )
        extended_pose_attention_mask = (1.0 - extended_pose_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=torch.float32 #next(self.parameters()).dtype
        )  
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        extended_audio_attention_mask = extended_audio_attention_mask.to(
            dtype=torch.float32#next(self.parameters()).dtype
        )
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(image_feat.size(0), image_feat.size(1), image_feat.size(1)).type_as(extended_image_attention_mask)         

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=torch.float32   #next(self.parameters()).dtype
        )  
        
        # repeat audio embedding to generate audio sequences
        audio_feat = audio_feat.unsqueeze(1).unsqueeze(1)
        audio_feat = audio_feat.repeat(1,image_feat.shape[1],image_feat.shape[2],1)
         
        encoded_layers_v, encoded_layers_p, encoded_layers_a, all_attention_mask = self.encoder(
            image_feat,
            pose_feat,
            audio_feat,
            extended_image_attention_mask,
            extended_pose_attention_mask,
            extended_audio_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        
        sequence_output_v = encoded_layers_v[-1]
        sequence_output_p = encoded_layers_p[-1]
        sequence_output_a = encoded_layers_a[-1]
           
        pooled_output_p = self.p_pooler(sequence_output_p)
        pooled_output_v = self.v_pooler(sequence_output_v)
        pooled_output_a = self.a_pooler(sequence_output_a)

        if not output_all_encoded_layers:
            encoded_layers_p = encoded_layers_p[-1]
            encoded_layers_v = encoded_layers_v[-1]
            encoded_layers_a = encoded_layers_a[-1]

        return encoded_layers_v, encoded_layers_p, encoded_layers_a, pooled_output_v, pooled_output_p, pooled_output_a, all_attention_mask


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.
       
        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.p_biattention_id = config.p_biattention_id
        self.a_biattention_id = config.a_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_p_layer = config.fixed_p_layer
        self.fixed_v_layer = config.fixed_v_layer
        self.fixed_a_layer = config.fixed_a_layer
        
        v_layer = BertImageLayer(config)
        p_layer = BertPoseLayer(config)
        a_layer = BertAudioLayer(config)
        
        connect_layer = BertConnectionLayer(config)

        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )
   
        self.p_layer = nn.ModuleList(
            [copy.deepcopy(p_layer) for _ in range(config.p_num_hidden_layers)]
        )

        self.a_layer = nn.ModuleList(
            [copy.deepcopy(a_layer) for _ in range(config.a_num_hidden_layers)]
        )

        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

    def forward(
        self,
        image_embedding,
        pose_embedding,
        audio_embedding,
        image_attention_mask,
        pose_attention_mask,
        audio_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
    ):
    
        v_start = 0
        p_start = 0
        a_start = 0
        count = 0
        all_encoder_layers_p = []
        all_encoder_layers_v = []
        all_encoder_layers_a = []

        all_attention_mask_p = []
        all_attnetion_mask_v = []
        all_attention_mask_a = []
        all_attention_mask_c = []
        
        batch_size, num_frames, num_regions, v_hidden_size = image_embedding.size()
        
        use_co_attention_mask = False
        for v_layer_id, p_layer_id, a_layer_id in zip(self.v_biattention_id, self.p_biattention_id, self.a_biattention_id):

            v_end = v_layer_id
            p_end = p_layer_id
            a_end = a_layer_id

            assert self.fixed_p_layer <= p_end
            assert self.fixed_v_layer <= v_end
            assert self.fixed_a_layer <= a_end
            
            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask)
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)
            
            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask)
                
                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(p_start, self.fixed_p_layer):
                with torch.no_grad():
                    pose_embedding, pose_attention_probs = self.p_layer[idx](pose_embedding, pose_attention_mask)
                    p_start = self.fixed_p_layer
                    if output_all_attention_masks:
                        all_attention_mask_p.append(pose_attention_probs)

            for idx in range(p_start, p_end):
                pose_embedding, pose_attention_probs = self.p_layer[idx](pose_embedding, pose_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_p.append(pose_attention_probs)

            for idx in range(a_start, self.fixed_a_layer):
                with torch.no_grad():
                    audio_embedding, audio_attention_probs = self.a_layer[idx](audio_embedding,audio_attention_mask )
                    a_start = self.fixed_a_layer
                    if output_all_attention_masks:
                        all_attention_mask_a.append(audio_attention_probs)

            for idx in range(a_start, a_end):
                audio_embedding, audio_attention_probs = self.a_layer[idx](audio_embedding, audio_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_a.append(audio_attention_probs)
      
            if self.with_coattention:
                # do the bi attention.
                image_embedding, pose_embedding, audio_embedding, co_attention_probs = self.c_layer[count](image_embedding, image_attention_mask, pose_embedding, pose_attention_mask, audio_embedding, audio_attention_mask, co_attention_mask, use_co_attention_mask)
                
                # use_co_attention_mask = False
                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            p_start = p_end
            a_start = a_end
            count += 1
            
            if output_all_encoded_layers:
                all_encoder_layers_p.append(pose_embedding)
                all_encoder_layers_v.append(image_embedding)
                all_encoder_layers_a.append(audio_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask)

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)
        
        for idx in range(p_start, len(self.p_layer)):
            pose_embedding, pose_attention_probs = self.p_layer[idx](pose_embedding, pose_attention_mask)

            if output_all_attention_masks:
                all_attention_mask_p.append(pose_attention_probs)

        for idx in range(a_start, len(self.a_layer)):
            audio_embedding, audio_attention_probs = self.a_layer[idx](audio_embedding, audio_attention_mask)

            if output_all_attention_masks:
                all_attention_mask_a.append(audio_attention_probs)
        
        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_v.append(image_embedding)
            all_encoder_layers_p.append(pose_embedding)
            all_encoder_layers_a.append(audio_embedding)

        return all_encoder_layers_v, all_encoder_layers_p, all_encoder_layers_a, (all_attnetion_mask_v, all_attention_mask_p, all_attention_mask_a, all_attention_mask_c)


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        B, T, S, F = hidden_states.size()
        hidden_states = hidden_states.view(B, T*S, F)
        attention_mask = attention_mask.view(B, 1, 1, T*S)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
       
        context_layer = context_layer.view(B, T, S, F) 
        return context_layer, attention_probs

class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

class BertPoseAttention(nn.Module):
    def __init__(self, config):
        super(BertPoseAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class BertPoseIntermediate(nn.Module):
    def __init__(self, config):
        super(BertPoseIntermediate, self).__init__()
        self.dense = nn.Linear(config.p_hidden_size, config.p_intermediate_size)
        if isinstance(config.p_hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.p_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.p_hidden_act]
        else:
            self.intermediate_act_fn = config.p_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertPoseOutput(nn.Module):
    def __init__(self, config):
        super(BertPoseOutput, self).__init__()
        self.dense = nn.Linear(config.p_intermediate_size, config.p_hidden_size)
        self.LayerNorm = BertLayerNorm(config.p_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.p_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPoseLayer(nn.Module):
    def __init__(self, config):
        super(BertPoseLayer, self).__init__()
        self.attention = BertPoseAttention(config)
        self.intermediate = BertPoseIntermediate(config)
        self.output = BertPoseOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

class BertAudioSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertAudioSelfAttention, self).__init__()
        if config.a_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.a_hidden_size / config.v_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.a_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.a_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.a_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        B, T, S, F = hidden_states.size()
        hidden_states = hidden_states.view(B, T*S, F)
        attention_mask = attention_mask.view(B, 1, 1, T*S)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.view(B, T, S, F)
        return context_layer, attention_probs

class BertAudioSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertAudioSelfOutput, self).__init__()
        self.dense = nn.Linear(config.a_hidden_size, config.a_hidden_size)
        self.LayerNorm = BertLayerNorm(config.a_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAudioAttention(nn.Module):
    def __init__(self, config):
        super(BertAudioAttention, self).__init__()
        self.self = BertAudioSelfAttention(config)
        self.output = BertAudioSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class BertAudioIntermediate(nn.Module):
    def __init__(self, config):
        super(BertAudioIntermediate, self).__init__()
        self.dense = nn.Linear(config.a_hidden_size, config.a_intermediate_size)
        if isinstance(config.a_hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.a_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.a_hidden_act]
        else:
            self.intermediate_act_fn = config.a_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertAudioOutput(nn.Module):
    def __init__(self, config):
        super(BertAudioOutput, self).__init__()
        self.dense = nn.Linear(config.a_intermediate_size, config.a_hidden_size)
        self.LayerNorm = BertLayerNorm(config.a_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.a_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAudioLayer(nn.Module):
    def __init__(self, config):
        super(BertAudioLayer, self).__init__()
        self.attention = BertAudioAttention(config)
        self.intermediate = BertAudioIntermediate(config)
        self.output = BertAudioOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.p_intermediate = BertPoseIntermediate(config)
        self.p_output = BertPoseOutput(config)

        self.a_intermediate = BertAudioIntermediate(config)
        self.a_output = BertAudioOutput(config)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, input_tensor3, attention_mask3, co_attention_mask=None, use_co_attention_mask=False):
 
        bi_output1, bi_output2, bi_output3, co_attention_probs = self.biattention(
            input_tensor1, attention_mask1, input_tensor2, attention_mask2, input_tensor3,attention_mask3, co_attention_mask, use_co_attention_mask
        ) 
        
        attention_output1, attention_output2, attention_output3 = self.biOutput(bi_output1, input_tensor1, bi_output2, input_tensor2, bi_output3, input_tensor3) 

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)
        
        intermediate_output2 = self.p_intermediate(attention_output2)
        layer_output2 = self.p_output(intermediate_output2, attention_output2)

        intermediate_output3 = self.a_intermediate(attention_output3)
        layer_output3 = self.a_output(intermediate_output3, attention_output3)

        return layer_output1, layer_output2,layer_output3, co_attention_probs

class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.all_head_size_kv = int(self.all_head_size/2)

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size_kv)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size_kv)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.p_hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.p_hidden_size, self.all_head_size_kv)
        self.value2 = nn.Linear(config.p_hidden_size, self.all_head_size_kv)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.v_attention_probs_dropout_prob)
 
        self.query3 = nn.Linear(config.a_hidden_size, self.all_head_size)
        self.key3 = nn.Linear(config.a_hidden_size, self.all_head_size_kv)
        self.value3 = nn.Linear(config.a_hidden_size, self.all_head_size_kv)

        self.dropout3 = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_kv(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            int(self.attention_head_size/2),
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, input_tensor3, attention_mask3, co_attention_mask=None, use_co_attention_mask=False):
        
        #combine number of frames and sequence together for easily computation
        B, T, S, F = input_tensor1.size()
        input_tensor1 = input_tensor1.view(B, T*S, -1)
        input_tensor2 = input_tensor2.view(B, T*S, -1)
        input_tensor3 = input_tensor3.view(B, T*S, -1)
        attention_mask1 = attention_mask1.view(B, 1, 1, T*S)
        attention_mask2 = attention_mask2.view(B, 1, 1, T*S)
        attention_mask3 = attention_mask3.view(B, 1, 1, T*S)

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)
       
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores_kv(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores_kv(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for pose input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores_kv(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores_kv(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        #for audio input:
        mixed_query_layer3 = self.query3(input_tensor3)
        mixed_key_layer3 = self.key3(input_tensor3)
        mixed_value_layer3 = self.value3(input_tensor3)

        query_layer3 = self.transpose_for_scores(mixed_query_layer3)
        key_layer3 = self.transpose_for_scores_kv(mixed_key_layer3)
        value_layer3 = self.transpose_for_scores_kv(mixed_value_layer3)

        ###pose attention score      
        attention_scores1 = torch.matmul(query_layer2, torch.cat((key_layer1,key_layer3),3).transpose(-1, -2)) 
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + torch.tensordot(attention_mask1, attention_mask3)

        if use_co_attention_mask:
            attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, torch.cat((value_layer1, value_layer3),3))
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        ###vision attention score
        attention_scores2 = torch.matmul(query_layer1, torch.cat((key_layer2, key_layer3),3).transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)


        if use_co_attention_mask:
            attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, torch.cat((value_layer2, value_layer3),3))
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
       
        #for audio
        attention_scores3 = torch.matmul(query_layer3, torch.cat((key_layer1, key_layer2),3).transpose(-1, -2))
        attention_scores3 = attention_scores3 / math.sqrt(self.attention_head_size)
        attention_scores3 = attention_scores3 + torch.tensordot(attention_mask1,attention_mask2)

        if use_co_attention_mask:
            attention_scores3 = attention_scores3 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs3 = nn.Softmax(dim=-1)(attention_scores3)
        attention_probs3 = self.dropout3(attention_probs3)
        
        context_layer3 = torch.matmul(attention_probs3, torch.cat((value_layer1, value_layer2),3))
        context_layer3 = context_layer3.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape3 = context_layer3.size()[:-2] + (self.all_head_size,)
        context_layer3 = context_layer3.view(*new_context_layer_shape3)
        #reshape back to original sequence 
        context_layer1 = context_layer1.view(B, T, S, -1)
        context_layer2 = context_layer2.view(B, T, S, -1)
        context_layer3 = context_layer3.view(B, T, S, -1)
        #vision, pose, audio
        return context_layer2, context_layer1, context_layer3, (attention_probs2, attention_probs1, attention_probs3)

class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.p_hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.p_hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.p_hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.dense3 = nn.Linear(config.bi_hidden_size, config.a_hidden_size)
        self.LayerNorm3 = BertLayerNorm(config.a_hidden_size, eps=1e-12)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense3 = nn.Linear(config.bi_hidden_size, config.a_hidden_size)
        self.q_dropout3 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2, hidden_states3, input_tensor3):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        context_state3 = self.dense3(hidden_states3)
        context_state3 = self.dropout3(context_state3)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        hidden_states3 = self.LayerNorm3(context_state3 + input_tensor3)

        return hidden_states1, hidden_states2, hidden_states3

class BertPosePooler(nn.Module):
    def __init__(self, config):
        super(BertPosePooler, self).__init__()
        self.dense = nn.Linear(config.p_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertAudioPooler(nn.Module):
    def __init__(self, config):
        super(BertAudioPooler, self).__init__()
        self.dense = nn.Linear(config.a_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ResnetDilated, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

        self.linear = nn.Linear(64,32)

        self.pose_embeddings = BertPoseEmbeddings()

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pose_feat, pool=True):
        (B, C, T, H, W) = x.size()
        #import ipdb; ipdb.set_trace()
        pose_feat = self.pose_embeddings(pose_feat)
        #pose_feat = pose_feat.permute(0,2,1).contiguous() 
        #pose_feat = F.adaptive_avg_pool1d(pose_feat,1)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)

        x = x.view(B, C)
        # fuse both vision and pose feat
        x = x.unsqueeze(1)
        x = x.repeat(1,pose_feat.shape[1],1)
        #pose_feat = pose_feat.view(B, C)
        x = torch.cat((x, pose_feat), dim=2)  #x * pose_feat
        x = self.linear(x)
   
        return x
