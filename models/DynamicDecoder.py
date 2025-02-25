import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers): # default: (256,256,4,3)
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, norm=None, return_intermediate=False, num_queries=100):
        super().__init__()

        self.d_model = 256
        decoder_layer = TransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, num_queries=num_queries)
               
        self.num_queries = 300
        hidden_dim = 256

        self.num_queries = num_queries
        
        self.num_classes = 91 
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_embed_reg = MLP(hidden_dim, hidden_dim, 4, 3)

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.bbox_embed = nn.Embedding(self.num_queries, 4)
        
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(self.d_model)
        self.return_intermediate = return_intermediate

    def forward(self, feature,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                bbox_pos: Optional[Tensor] = None):
        
        bs, c, h, w = feature.shape
        output = self.query_embed.weight
        box = self.bbox_embed.weight
        output = output.unsqueeze(1).repeat(1, bs, 1) # 300 x bs x 256
        box = box.unsqueeze(1).repeat(1, bs, 1) # 300 x bs x 4
        
        intermediate_obj_emb = []
        intermediate_bbox = []

        for layer in self.layers:
            output, box = layer(output, feature, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, bbox_pos=box)
            #print(output.shape, box.shape)
            
            if self.return_intermediate:
                intermediate_obj_emb.append(self.norm(output).permute(1, 0, 2))
                intermediate_bbox.append(box.permute(1, 0, 2))
                 
        if self.norm is not None:
            output = self.norm(output).permute(1, 0, 2)
            box = box.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate_obj_emb.pop()
                intermediate_obj_emb.append(output)
                intermediate_bbox.pop()
                intermediate_bbox.append(box)
        
        #print(torch.stack(intermediate_obj_emb).shape) # 6, bs, 300, 256
        #print(torch.stack(intermediate_bbox).shape)
                
        if self.return_intermediate:
            return self.class_embed(torch.stack(intermediate_obj_emb)), torch.stack(intermediate_bbox).sigmoid()
        
        return self.class_embed(output.unsqueeze(0)), box.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, nhead=4, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False, num_queries=100):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model+4, nhead, dropout=dropout)
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        #self.fc_kernel = nn.Linear(300, d_model)
        self.fc_kernel = nn.Linear(d_model, d_model**2)

        self.num_queries=num_queries
        self.FFN = MLP(49, 98, 1, 3)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = MLP(d_model, d_model, 4, 3)
        self.norm2 = nn.LayerNorm(4)
        self.dropout2 = nn.Dropout(dropout)

        self.conv_norm = nn.BatchNorm2d(256)
        self.activation = 'relu'
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return torch.cat((tensor, pos), dim=-1)

    def forward_post(self, object_embed, feature,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    bbox_pos: Optional[Tensor] = None):
                    
        # feature: NCHW
        # bbox_pos: 300 x bs x 4
        # object_embed: previous output of decoder layer 300 x bs x 256

        bs, c, h, w = feature.shape

        bbox_sigmoid = bbox_pos.sigmoid()
        
        q = k = self.with_pos_embed(object_embed, bbox_pos)
        
        p3d = (0, 4)
        v = F.pad(object_embed, p3d, "constant", 0)
        
        
        #q = k = v =  object_embed
        object_embed2 = self.self_attn(q, k, v)[0]
        
        #object_embed = object_embed + self.dropout1(object_embed2)
        object_embed = object_embed + self.dropout1(object_embed2)[:,:,:256]
        object_embed = self.norm1(object_embed) # 300 x bs x 256
        
        box = box_cxcywh_to_xyxy(bbox_sigmoid)
        box_list = [box[:,i] for i in range(bs)]
        out = torchvision.ops.roi_pool(feature, box_list, output_size=7) # bs*300, 256, 7, 7
        
        num, C, H, W = out.shape
        #print("Out: ", out.shape)
        out = out.view(bs, int(num/bs), C, H, W) 
        #print("Out: ", out.shape)  # bs, 300, 256, 7, 7
        
        Q_star = object_embed.permute(1, 0, 2) # bs, 300, 256
        
        dynamic_kernel = self.fc_kernel(Q_star) # [bs, 300, 256**2]
        dynamic_kernel = dynamic_kernel.view(bs, self.num_queries, 256, 256) 
        
        Q_F = []  # 存储最终的 Query 特征
        for i in range(bs):  # 遍历 batch
            Q_FF = []  # 存储当前 batch 内的 300 个 Query 结果
            for j in range(self.num_queries):  # 遍历 300 个 Query
                k = dynamic_kernel[i, j, :]  # 取出第 i 个 batch，第 j 个 query 的动态卷积核
                k = torch.unsqueeze(k, -1)  # (256, 256, 1)
                k = torch.unsqueeze(k, -1)  # (256, 256, 1, 1) -> 卷积核需要 4D 形状 (out_ch, in_ch, h, w)
        
                feat_in = out[i, j].unsqueeze(0)  # 取出 RoI 特征 (256, 7, 7) -> 添加 batch 维度变成 (1, 256, 7, 7)
        
                out_temp = self.conv_norm(F.conv2d(feat_in, k, padding='same'))  
                # 使用动态卷积核 k 对特征 feat_in 进行 2D 卷积
                # conv2d 输入: (1, 256, 7, 7)
                # conv2d 卷积核: (256, 256, 1, 1)
                # 输出形状: (1, 256, 7, 7)
                # `conv_norm` 进行归一化
                
                Q_FF.append(out_temp.squeeze(0))  # 去掉 batch 维度，得到 (256, 7, 7)
        
            Q_F.append(torch.stack(Q_FF))  # 组装 batch 维度，变成 (300, 256, 7, 7)
            
            ######--------------------------------------------------------------
            #print(Q_F[0].shape)
        Q_F = torch.stack(Q_F) # bs, 300, 256, 7, 7
        #print("Q_F: ", Q_F.shape)
        Q_F = Q_F.view(bs, self.num_queries, 256, 49)
        #print("Q_F: ", Q_F.shape)
        ######--------------------------------------------------------------
        
        #print("Q_star: ", Q_star.shape)
        #print("dynamic_kernel: ", dynamic_kernel.shape)
        
        #Q_star = object_embed.permute(1, 2, 0).reshape(bs, 256, int(num/bs)) # bs, 256, 300
        #print("Q_star: ", Q_star.shape)
        #dynamic_kernel = self.fc_kernel(Q_star) # [bs, 256, 256]  
        #dynamic_kernel = dynamic_kernel.view(bs, 256, 256)
        #print("object_embed: ", object_embed.shape)
        '''
        Q_F = []
        for i in range(bs):
            k = dynamic_kernel[i]
            k = torch.unsqueeze(k, -1)
            k = torch.unsqueeze(k, -1)
            #print("k: ", k.shape)
            #print(out[i].shape)
            Q_F.append(self.conv_norm(F.conv2d(out[i], k, padding='same')) + out[i]) # [(300,256,7,7)]
            ######--------------------------------------------------------------
            #print(Q_F[0].shape)

        Q_F = torch.stack(Q_F) # bs, 300, 256, 7, 7
        #print("Q_F: ", Q_F.shape)
        Q_F = Q_F.view(bs, 300, 256, 49)
        #print("Q_F: ", Q_F.shape)
        '''
        
        new_Q = self.FFN(Q_F).squeeze(-1).permute(1, 0, 2) # 300, bs, 256
        ######--------------------------------------------------------------
        new_B = self.linear2(new_Q) # 300, bs, 4 (no sigmoid)
        
        #print("new_Q: ", new_Q.shape)
        #print("new_B: ", new_B.shape)

        return new_Q, new_B
        
        
    def forward(self, object_embed, feature,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                bbox_pos: Optional[Tensor] = None):

        return self.forward_post(object_embed, feature, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, bbox_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
