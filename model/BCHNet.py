r""" The framework of BCHNet for CD-FSS """

from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models import vgg
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from .base.frequency_filter import Frequency_filter


class BCHNetwork(nn.Module):
    def __init__(self, backbone):
        super(BCHNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone

        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'vgg16':
            print("Sorry. You need to redesign it by yourself.")
            # self.backbone = vgg.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            # self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            # self.extract_feats = extract_feat_vgg
            # nbottlenecks = [2, 2, 3, 3, 3, 1]

        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Frequency domain filter
        self.FDF = Frequency_filter(shape=(1, 2048, 13, 13))

    def forward(self, query_img, support_img, support_mask):

        # ################################ Features Extracting #########################################################
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        fg_support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())
        # ################################ Features Extracting #########################################################

        # ################################ Query Self-similarity Matching ##############################################
        sim_mat = self.similarity(query_feats, prototypes_f, prototypes_b)
        P_f1, P_b2 = self.SSP_func(query_feats, sim_mat)
        M_self = self.self_similarity(query_feats, P_f1)
        Pf_self = self.masked_average_pooling(query_feats, M_self)
        mask_rough = self.query_similarity_func(query_feats, Pf_self, P_b2)
        M_rough = []  # store rough mask for loss computation
        for idx, mask_f in enumerate(mask_rough):
            mask_b = 1 - mask_f
            cat_mask = torch.cat((mask_b.unsqueeze(1), mask_f.unsqueeze(1)), dim=1)
            M_rough.append(cat_mask)
        fg_query_feats = self.mask_query_feature(query_feats, mask_rough)
        # ############################### Query Self-similar Matching ##################################################

        # ################################ Frequency Domain Filter #####################################################
        query_feats = self.filter(query_feats)
        fg_support_feats = self.filter(fg_support_feats)
        support_feats = self.filter(support_feats)
        fg_query_feats = self.filter(fg_query_feats)
        # ################################ Frequency Domain Filter ####################################################

        # ############################### Bidirectional Hypercorrelation Construction ##################################
        corr_1 = Correlation.multilayer_correlation(query_feats, fg_support_feats, self.stack_ids)
        corr_2 = Correlation.multilayer_correlation(support_feats, fg_query_feats, self.stack_ids)
        logit_mask_1 = self.hpn_learner(corr_1)
        logit_mask_1 = F.interpolate(logit_mask_1, support_img.size()[2:], mode='bilinear', align_corners=True)
        logit_mask_2 = self.hpn_learner(corr_2)
        logit_mask_2 = F.interpolate(logit_mask_2, support_img.size()[2:], mode='bilinear', align_corners=True)
        # ############################### Bidirectional Hypercorrelation Construction ##################################

        return logit_mask_1, logit_mask_2, M_rough

    def mask_feature(self, features, support_mask):

        # feature      [bsz, C, H, W]
        # support_mask [bsz, h, w]

        eps = 1e-6
        prototypes_f = []
        prototypes_b = []
        # bg_features = []
        fg_features = []
        for idx, feature in enumerate(features):

            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            bg_mask = 1 - mask

            feature_f = feature * mask
            feature_b = feature * bg_mask

            # prototype
            proto_f = feature_f.sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_f.append(proto_f)

            proto_b = feature_b.sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            prototypes_b.append(proto_b)

            fg_features.append(feature_f)
            # bg_features.append(feature_b)

        return fg_features, prototypes_f, prototypes_b

    def similarity(self, features, fg_proto, bg_proto):

        out = []
        for idx, feature in enumerate(features):
            # Calculate cosine similarity with foreground prototype and background prototype respectively
            similarity_fg = F.cosine_similarity(features[idx], fg_proto[idx].unsqueeze(2).unsqueeze(3), dim=1)     # [bsz, h, w]
            similarity_bg = F.cosine_similarity(features[idx], bg_proto[idx].unsqueeze(2).unsqueeze(3), dim=1)     # [bsz, h, w]
            out_cat = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0  # [bsz, 2, h, w]
            out.append(out_cat)                                                                                    # len = 13

        return out

    def SSP_func(self, features, sim_mat):

        Pq_f = []
        Pq_sspb = []
        for i, feature_q in enumerate(features):

            bs = features[i].shape[0]  # [bsz, c, h, w]  bs = bsz
            ch = features[i].shape[1]  # [bsz, c, h, w]  ch = ch
            # h  = features[i].shape[2]  # [bsz, c, h, w]  h = h
            # w  = features[i].shape[3]  # [bsz, c, h, w]  w = w

            pred_1 = sim_mat[i].softmax(1)  # [bsz, c, h, w]
            pred_1 = pred_1.view(bs, 2, -1)  # Reshape into the shape of (bsz, 2, h * w)

            pred_fg = pred_1[:, 1]  # Foreground similarity
            pred_bg = pred_1[:, 0]  # Background similarity

            fg_ls = []        # Store significant  foreground prototypes
            bg_local_ls = []  # Store self-support foreground prototypes

            for epi in range(bs):
                fg_thres = 0.7  # foreground threshold t1
                bg_thres = 0.6  # background threshold t2
                cur_feat = feature_q[epi].view(ch, -1)  # Reshape the query features into the shape of (ch, h * w)
                f_h, f_w = feature_q[epi].shape[-2:]  # Obtain height and width

                # FC layer prototype
                if i <= 3:
                    k = 12
                elif i <= 9:
                    k = 6
                elif i <= 12:
                    k = 3

                # extract foreground pixels
                if (pred_fg[epi] > fg_thres).sum() > 0:
                    fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]  # shape: (ch, num_of_fg_pixels)
                else:
                    fg_feat = cur_feat[:, torch.topk(pred_fg[epi], k).indices]

                # extract background pixels
                if (pred_bg[epi] > bg_thres).sum() > 0:
                    bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]  # shape: (ch, num_of_bg_pixels)
                else:
                    bg_feat = cur_feat[:, torch.topk(pred_bg[epi], k).indices]

                # Get significant foreground prototype
                fg_proto = fg_feat.mean(-1)
                fg_ls.append(fg_proto.unsqueeze(0))

                # Get self-support background prototype
                bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)     # Normalized background features
                cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)  # Normalize the original features
                cur_feat_norm_t = cur_feat_norm.t()  # transpose, from (ch, h*w) to (h*w, ch)
                bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # to obtain the background similarity matrix
                bg_sim = bg_sim.softmax(-1)
                bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
                bg_proto_local = bg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0)  # self-support background prototype
                bg_local_ls.append(bg_proto_local)

            # save significant foreground prototype & self-support background prototype
            Pq_f_i = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
            Pq_sspb_i = torch.cat(bg_local_ls, 0)
            Pq_f.append(Pq_f_i)        # len=13: [bsz, c, 1, 1]
            Pq_sspb.append(Pq_sspb_i)  # len=13: [bsz, c, h, w]

        return Pq_f, Pq_sspb

    def self_similarity(self, features, fg_proto):

        M_self =[]
        for i, feature_q in enumerate(features):
            # Calculate similarity with significant foreground prototypes
            similarity_fg = F.cosine_similarity(feature_q, fg_proto[i], dim=1)  # similarity_fg.shape: [bsz, h, w]
            threshold = 0.6
            foreground_mask = (similarity_fg >= threshold).float()
            M_self_fg = foreground_mask      # [bsz, h, w]
            M_self_bg = 1 - foreground_mask  # [bsz, h, w]
            M_self_i = torch.cat((M_self_bg.unsqueeze(1), M_self_fg.unsqueeze(1)), dim=1)  # [bsz, 2, h, w]
            M_self.append(M_self_i)  # len = 13: [bsz, 2, h, w]  fg & bg

        return M_self

    def masked_average_pooling(self, features, masks):

        prototypes_self_f = []  # total 13 protos for query foreground
        eps = 1e-6
        for idx, mask in enumerate(masks):
            pred_mask = mask.argmax(dim=1)               # M_self     [bsz, h, w]
            pred_mask = pred_mask.unsqueeze(1).float()   # foreground [bsz, 1, h, w]
            fg_feature = features[idx] * pred_mask
            proto_f = fg_feature.sum((2, 3))
            label_sum = pred_mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)  # [bsz, c]
            prototypes_self_f.append(proto_f.unsqueeze(-1).unsqueeze(-1))  # len=13: [bsz, c, h, w]
        return prototypes_self_f

    def query_similarity_func(self, features, fg_proto, bg_proto):

        out = []
        for idx, feature in enumerate(features):
            similarity_fg = F.cosine_similarity(features[idx], fg_proto[idx], dim=1)  # [bsz, h, w]
            similarity_bg = F.cosine_similarity(features[idx], bg_proto[idx], dim=1)  # [bsz, h, w]
            out_cat = torch.cat((similarity_bg.unsqueeze(1), similarity_fg.unsqueeze(1)), dim=1) * 10.0  # [bsz, 2, h, w]
            out_cat = out_cat.argmax(dim=1)  # [bsz, h, w]
            out.append(out_cat)  # len=13: [bsz, h, w]

        return out

    def mask_query_feature(self, features, rough_masks):

        fg_features = []

        for idx, feature in enumerate(features):
            mask = rough_masks[idx].unsqueeze(1)  # [bsz, 1, H, W]
            feature_f = feature * mask            # [bsz, C, H, W] = [bsz, C, H, W]*[bsz, 1, H, W]
            fg_features.append(feature_f)

        return fg_features

    def filter(self, feats):
        apm_feats = []
        for idx, feature in enumerate(feats):
            if idx <= 9:
                apm_feats.append(feature)
            elif idx <= 12:
                x = self.FDF(feature)
                apm_feats.append(x)
        return apm_feats

    def predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_orig = []
        bg_logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask, bg_logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            bg_logit_mask_orig.append(bg_logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig, bg_logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig, bg_logit_mask_orig

    def final_predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask = self.final_predict(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig

    def final_predict(self, query_img, support_img, support_mask):  # Final one-way prediction

        # Features Extracting
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                               self.lids)
        fg_support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())

        # Frequency Domain Filter
        query_feats = self.filter(query_feats)
        fg_support_feats = self.filter(fg_support_feats)

        # Hypercorrelation Construction
        corr_1 = Correlation.multilayer_correlation(query_feats, fg_support_feats, self.stack_ids)
        logit_mask_1 = self.hpn_learner(corr_1)
        logit_mask_1 = F.interpolate(logit_mask_1, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask_1
    
    def predict_mask_nshot_support(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_orig = []

        for s_idx in range(nshot):
            _, logit_mask, _ = self(batch['support_imgs'][:, s_idx], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)
    
    def compute_objective_finetuning(self, logit_mask, gt_mask, nshot):
        # logit_mask [nshot, bsz, c, h, w]
        # gt_mask [bsz, nshot, h, w]
        loss = 0.0
        for idx in range(nshot):
            bsz = gt_mask.shape[0]
            loss += self.cross_entropy_loss(logit_mask[idx].view(bsz, 2, -1), gt_mask[:,idx].view(bsz, -1).long())
        return loss/nshot

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def pred_mask_loss(self, pred_mask, gt):
        # pred_mask [layernum, bsz, 2, h, w]
        # gt [bsz, h, w]
        loss =  0.0
        for mask in pred_mask:
            gt_mask = F.interpolate(gt.unsqueeze(1).float(), mask.size()[2:], mode='bilinear', align_corners=True)
            loss = loss + self.compute_objective(mask.float(), gt_mask)
        loss = loss / 13.0
        return loss