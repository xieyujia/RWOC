#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:59:33 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:48:20 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:28:09 2020

@author: yujia
"""


# ==========================================================================
#
# This file is a part of implementation for paper:
# DeepMOT: A Differentiable Framework for Training Multiple Object Trackers.
# This contribution is headed by Perception research team, INRIA.
#
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
#
# ===========================================================================

import os
import random
import shutil
import argparse
import torch.optim as optim
import torch
import numpy as np
from models.siamrpn import SiamRPNvot
from models.sinkhorn import Sinkhorn_custom
from os.path import realpath, dirname
from tensorboardX import SummaryWriter
from utils.sot_utils import *
from utils.loss import *
from models.DAN import build_sst
from utils.DAN_utils import TrackUtil
from utils.io_utils import read_txt_detV2
from utils.mot_utils import tracking_birth_death
from utils.tracking_config import tracking_config
from utils.box_utils import *


def main(args, sot_tracker, deepMunkres, sst, optimizer, loss_writer):
    """
    train a sot to perform MOT using DeepMOT Loss
    :param args: parameters, argparse
    :param sot_tracker: single object tracker, torch network
    :param deepMunkres: deep Hungarian Net, torch network
    :param optimizer: training optimizer, torch optim
    :param mota_writer: record MOTA loss, tensorboardX writer
    :param motp_writer: record MOTP loss, tensorboardX writer
    :param clasf_writer: record classification loss, tensorboardX writer
    """

    iterations = 0
    chunks = {}
    old_loss = 100
    for epoch in range(args.epochs):
        pth = args.data_root + args.dataset + '/train/'
        nmspth = args.dets_path+args.dataset+'/train/'
        print("training...")
        print("Dataset from: ", pth)
        videos = os.listdir(pth)
        random.shuffle(videos)

        for vname in videos:
            if "flip" in vname or "rot" in vname or 'DPM' not in vname:
                continue

            print("***************************************************************")
            print(vname)
            print("***************************************************************")
            
            #####################################################################
            # load MOT configuration #
            to_refine, to_combine, DAN_th, death_count, birth_wait, loose_assignment, \
            case1_interpolate, interpolate_flag, CMC, birth_iou = tracking_config(vname, args.dataset)

            print("tracking video: ")
            print(vname)

            track_init = []  # near online track init

            # load image paths #
            imgs_path = pth + vname + '/img1/'
            imgs = sorted(os.listdir(imgs_path))
            
            # cut video into small sequences of 100 frames
            if epoch == 0:
                tem = []
                for i in range(0, len(imgs), args.seq_len):
                    tem.append([i, imgs[i:i + args.seq_len]])
                chunks[vname] = tem + []
                del tem

            random.shuffle(chunks[vname])

            # load clean detections #
            if os.path.exists(nmspth + vname + '/det/det.npy'):
                frames_det = np.load(nmspth + vname + '/det/det.npy', allow_pickle=True).item()
            else:
                frames_det = read_txt_detV2(nmspth + vname + '/det/det.txt')

            if len(frames_det.keys()) == 0:
                print("cannot load detections")
                break

            

            # for a small sequence

            for i in range(len(chunks[vname])):
                first_idx, subset = chunks[vname][i]

                # tracking #

                # previous numpy frame
                img_prev = None
    
                # track id counter
                count_ids = 0
    
                # bbox_track = {frame_id: [[bbox], [bbox], [bbox]..]} dict of torch tensor with shape
                # [num_tracks, 4=(x1,y1,x2,y2)]
                bbox_track = dict()
    
                # id track = ordered [hypo_id1, hypo_id2, hypo_id3...] corresponding to bbox_track
                # of current frame, torch tensor [num_tracks]
                id_track = list()
    
                # states = {track_id: state, ...}
                states = dict()
    
                # previous frame id
                prev_frame_id = 0
    
                # birth candidates, {frames_id:[to birth id(=det_index)...], ...}
                birth_candidates = dict()
    
                # death candidates, {track_id:num_times_track_lost,lost_track ...}
                death_candidates = dict()
    
                # collect_prev_pos = {trackId:[postion_before_lost=[x1,y1,x2,y2], track_appearance_features,
                # matched_count, matched_det_collector(frame, det_id),
                # track_box_collector=[[frameid,[x,y,x,y]],...],'active' or 'inactive', velocity, inactive_pre_pos]}
                collect_prev_pos = dict()
    
                bbox_track[prev_frame_id] = None
    
                to_interpolate = dict()
                w_matrix = None

                for frameid, im_pth in enumerate(subset):
                    distance_matrix = 0
                    frameid += first_idx
                    
                #####################################################################
                    img_curr = cv2.imread(os.path.join(imgs_path, im_pth))
                    h, w, _ = img_curr.shape
    
                    # tracking for current frame #
    
                    # having active tracks
                    if len(states) > 0:
                        tmp = []
                        im_prev_features = TrackUtil.convert_image(img_prev.copy())
    
                        # calculate affine transformation for current frame and previous frame
                        if img_prev is not None and CMC:
                            w_matrix = getWarpMatrix(img_curr, img_prev)
    
                        # FOR every track in PREVIOUS frame
                        for key, state_curr in states.items():
                            # center position at frame t-1
                            prev_pos = state_curr['target_pos'].copy()
                            prev_size = state_curr['target_sz'].copy()
    
                            prev_xyxy = [prev_pos[0] - 0.5 * prev_size[0],
                                         prev_pos[1] - 0.5 * prev_size[1],
                                         prev_pos[0] + 0.5 * prev_size[0],
                                         prev_pos[1] + 0.5 * prev_size[1]]
    
                            if state_curr['gt_id'] not in collect_prev_pos.keys():
    
                                # extract image features by DAN
                                prev_xywh = [prev_pos[0] - 0.5 * prev_size[0], prev_pos[1] - 0.5 * prev_size[1],
                                             prev_size[0], prev_size[1]]
                                prev_xywh = np.array([prev_xywh], dtype=np.float32)
                                prev_xywh[:, [0, 2]] /= float(w)
                                prev_xywh[:, [1, 3]] /= float(h)
                                track_norm_center = TrackUtil.convert_detection(prev_xywh)
    
                                tracks_features = sst.forward_feature_extracter(im_prev_features,
                                                                                track_norm_center).detach_()
    
                                collect_prev_pos[state_curr['gt_id']] = [[[frameid-1, np.array(prev_xyxy)]],
                                                                         [[frameid-1, tracks_features.clone()]], 0,
                                                                         list(), list(), 'active', [0.0, -1.0, -1.0],
                                                                         np.zeros((4))-1]
                                del tracks_features
    
                            elif collect_prev_pos[state_curr['gt_id']][5] == 'active':
    
                                # extract image features by DAN
                                prev_xywh = [prev_pos[0] - 0.5 * prev_size[0], prev_pos[1] - 0.5 * prev_size[1],
                                             prev_size[0], prev_size[1]]
                                prev_xywh = np.array([prev_xywh], dtype=np.float32)
                                prev_xywh[:, [0, 2]] /= float(w)
                                prev_xywh[:, [1, 3]] /= float(h)
                                track_norm_center = TrackUtil.convert_detection(prev_xywh)
                                tracks_features = sst.forward_feature_extracter(im_prev_features,
                                                                                track_norm_center).detach_()
    
                                # update positions and appearance features of active track
                                collect_prev_pos[state_curr['gt_id']][0].append([frameid-1, np.array(prev_xyxy)])
    
                                # only keep the latest 10 active positions (used for estimating velocity for interpolations)
                                if len(collect_prev_pos[state_curr['gt_id']][0]) > 10:
                                    collect_prev_pos[state_curr['gt_id']][0].pop(0)
    
                                # only keep the latest 3 appearance features (used for recovering invisible tracks)
                                collect_prev_pos[state_curr['gt_id']][1].append([frameid-1, tracks_features.clone()])
                                if len(collect_prev_pos[state_curr['gt_id']][1]) > 3:
                                    collect_prev_pos[state_curr['gt_id']][1].pop(0)
                                del tracks_features
    
                                # remove pre_lost_pos when a track is recovered
                                collect_prev_pos[state_curr['gt_id']][7] = np.zeros((4))-1
    
                                # update velocity during active mode if we have 10 (might be not consecutive) positions
                                if len(collect_prev_pos[state_curr['gt_id']][0]) == 10:
                                    avg_h = 0.0
                                    avg_w = 0.0
                                    for f, pos in collect_prev_pos[state_curr['gt_id']][0]:
                                        avg_h += (pos[3] - pos[1])
                                        avg_w += (pos[2] - pos[0])
                                    avg_h /= len(collect_prev_pos[state_curr['gt_id']][0])
                                    avg_w /= len(collect_prev_pos[state_curr['gt_id']][0])
                                    last_t, last_pos = collect_prev_pos[state_curr['gt_id']][0][-1]
                                    first_t, first_pos = collect_prev_pos[state_curr['gt_id']][0][0]
                                    # center point
                                    first_pos_center = np.array([0.5 * (first_pos[0] + first_pos[2]),
                                                                 0.5 * (first_pos[1] + first_pos[3])])
                                    last_pos_center = np.array([0.5 * (last_pos[0]+last_pos[2]),
                                                                0.5 * (last_pos[1]+last_pos[3])])
                                    velocity = (last_pos_center - first_pos_center)/(last_t-first_t)
                                    collect_prev_pos[state_curr['gt_id']][6] = [velocity, avg_h, avg_w]
                                    collect_prev_pos[state_curr['gt_id']][0] = [collect_prev_pos[state_curr['gt_id']][0][-1]]
    
                            else:
                                # inactive mode, do nothing
                                pass
                            #print(state_curr)
                            target_pos, target_sz, state_curr, _ = SiamRPN_track(state_curr, img_curr.copy(), sot_tracker,
                                                                                train=True, CMC=(img_prev is not None and CMC),
                                                                                 prev_xyxy=prev_xyxy, w_matrix=w_matrix)
                            tmp.append(torch.stack([target_pos[0] - target_sz[0]*0.5,
                                                    target_pos[1] - target_sz[1]*0.5,
                                                    target_pos[0] + target_sz[0]*0.5,
                                                    target_pos[1] + target_sz[1]*0.5], dim=0).unsqueeze(0))
                            del _
                            del target_pos
                            del target_sz
                            torch.cuda.empty_cache()
    
                        bbox_track[frameid] = torch.cat(tmp, dim=0)
                        
                         # get distance matrix tracks-gts #
                        _, distance_matrix = make_single_matrix_torchV2_fast(frames_det[str(frameid + 1)],
                                                                                  bbox_track[frameid], h, w)
                        # get output from DHN, i.e. assignment matrix #
                        output_track_gt = deepMunkres(distance_matrix)
                        loss = torch.sum(output_track_gt*distance_matrix)
                        #loss = torch.sum(tmp[0])
                        # loss backward and update weights
                        sot_tracker.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # save best model #
                        if (iterations + 1) % args.save_freq == 0 and old_loss > loss.item():
                            old_loss = float(loss.item())
                            print("best model is saved into:", args.save_path +
                                  "best_model_" + str(epoch) + ".pth")
    
                            torch.save(sot_tracker.state_dict(),
                                       args.save_path+"best_model_" + str(epoch) + ".pth")
    
                        # print results #
                        if (iterations + 1) % args.print_freq == 0:
                            print('Epoch: [{}] Iterations: [{}]\tLoss {:.4f}'.format(epoch, iterations, float(loss.item())))
    
                            loss_writer.add_scalar('Loss', loss.item(), iterations)
    
                            # save model #
                            if (iterations + 1) % (args.save_freq * 20) == 0:
                                print("model is saved into:", args.save_path +
                                      "model_" + str(epoch) + ".pth")
    
                                torch.save(sot_tracker.state_dict(), args.save_path + "model_" + str(epoch) + ".pth")
    
                        iterations += 1
    
                        # clean up
                        del output_track_gt
                        del distance_matrix
                        del bbox_track[prev_frame_id]
                        del tmp
                        torch.cuda.empty_cache()
    
                    else:
                        # having no tracks
                        bbox_track[frameid] = None
    
                    # refine and calculate "distance" (actually, iou) matrix #
                    if str(frameid + 1) in frames_det.keys():
                        distance = []
                        if bbox_track[frameid] is not None:
                            bboxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                            for bbox in bboxes:
                                IOU = bb_fast_IOU_v1(bbox, frames_det[str(frameid + 1)])
                                distance.append(IOU.tolist())
                            distance = np.vstack(distance)
    
                            # refine tracks bboxes with dets if iou > 0.6
                            if to_combine:
                                del bboxes
                                # mix dets and tracks boxes
                                bbox_track[frameid] = mix_track_detV2(
                                    torch.FloatTensor(distance).cuda(),
                                    torch.FloatTensor(frames_det[str(frameid + 1)]).cuda(), bbox_track[frameid])
    
                                boxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                                for idx, [key, state] in enumerate(states.items()):
                                    # print(idx, key, state['gt_id'])
                                    box = boxes[idx]
                                    state['target_pos'] = np.array([0.5*(box[2] + box[0]), 0.5*(box[3] + box[1])])
                                    state['target_sz'] = np.array([box[2] - box[0], box[3] - box[1]])
                                    states[key] = state
                                distance = []
                                bboxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                                for bbox in bboxes:
                                    IOU = bb_fast_IOU_v1(bbox, frames_det[str(frameid + 1)])
                                    distance.append(IOU.tolist())
                                distance = np.vstack(distance)
    
                        # no tracks
                        else:
                            distance = np.array(distance)
    
                        # birth and death process, no need to be differentiable #
                        if bbox_track[frameid] is not None:
                            bbox_track[frameid] = bbox_track[frameid].detach()
                        bbox_track[frameid], count_ids = \
                            tracking_birth_death(distance, bbox_track[frameid], frames_det, img_curr,
                                                 id_track, count_ids, frameid, birth_candidates, track_init,
                                                 death_candidates, states, sot_tracker, collect_prev_pos, sst, th=0.5,
                                                 birth_iou=birth_iou, to_refine=to_refine, DAN_th=DAN_th,
                                                 death_count=death_count, birth_wait=birth_wait,
                                                 to_interpolate=to_interpolate, interpolate_flag=interpolate_flag,
                                                 loose_assignment=loose_assignment, case1_interpolate=case1_interpolate)
    
                        del distance
                                    
    
                    else:
                        print("no detections! all tracks killed.")
                        bbox_track[frameid] = None
                        id_track = list()
                        states = dict()
                        death_candidates = dict()
                        collect_prev_pos = dict()
    
                    img_prev = img_curr.copy()
                    prev_frame_id = frameid
                    torch.cuda.empty_cache()


if __name__ == '__main__':

    # init parameters #
    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--dataset', dest='dataset', default="mot17", help='dataset name')

    parser.add_argument('--logs', dest='logs', default=curr_path + '/logs/',
                        help='logs path')

    parser.add_argument('--data_root', dest='data_root', default= curr_path + '/data/',
                        help='dataset root path')

    parser.add_argument('--models_root', dest='models_root',
                        default=curr_path + '/pretrained/',
                        help='pretrained models root path')

    # BiRNN configs

    parser.add_argument('--element_dim', dest='element_dim', default=1, type=int, help='element_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='hidden_dim')
    parser.add_argument('--target_size', dest='target_size', default=1, type=int, help='target_size')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--bidrectional', dest='bidrectional', default=True, type=bool, help='bidrectional')

    # training configs
    parser.add_argument('--is_cuda', dest='is_cuda', default=True, type=bool, help='use cuda?')
    parser.add_argument('--seq_len', dest='seq_len', default=100, type=int, help='small sequence length')
    parser.add_argument('--epochs', dest='epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--old_lr', dest='old_lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--print_freq', dest='print_freq', default=200, type=int, help='print loss frequency')
    parser.add_argument('--ref_freq', dest='ref_freq', default=10, type=int, help='update reference images frequency')
    parser.add_argument('--smax_scale', dest='smax_scale', default=10.0, type=float, help='softmax scaling factor')
    parser.add_argument('--save_freq', dest='save_freq', default=20, type=int, help='save model weights frequency')
    parser.add_argument('--save_path', dest='save_path', default=curr_path + '/saved_models/', help='save_path')
    parser.add_argument('--dets_path', dest='dets_path', default=curr_path + '/clean_detections/',
                        help='detections root path')
    parser.add_argument('--num_repeats', dest='num_repeats', default=3, type=int,
                        help='train a frame for how many times')
    
    
    parser.add_argument('--method', type=str, default='sinkhorn_robust',
                        help='nn | sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='entropy regularization coefficient, used for Sinkhorn')
    parser.add_argument('--max_inner_iter', type=int, default=200,
                        help='inner iteration number, used for Sinkhorn')
    parser.add_argument('--rho1', type=float, default=1e-3,
                        help='relaxition for the first marginal')
    parser.add_argument('--rho2', type=float, default=1e-3,
                        help='relaxition for the second marginal')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='grad for projected gradient descent for robust OT')

    args = parser.parse_args()

    # init sot tracker #
    sot_tracker = SiamRPNvot()
    print("loading SOT from: ", args.models_root )
    sot_tracker.load_state_dict(torch.load(args.models_root ))

    # freeze first three conv layers (feature extraction layers)
    cntr = 0
    to_freeze = [0, 1, 4, 5, 8, 9]
    for child in sot_tracker.children():
        cntr += 1
        if cntr <= 1:
            for i, param in enumerate(child.parameters()):
                # print(i, param.shape)
                if i in to_freeze:
                    param.requires_grad = False

    # init optimizer #
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, sot_tracker.parameters()), lr=args.old_lr)

    # init munkres net #
#    print("loading DHN from: ", args.models_root + "DHN.pth")
#    deepMunkres = Munkrs(element_dim=args.element_dim, hidden_dim=args.hidden_dim, target_size=args.target_size,
#                         biDirenction=args.bidrectional, minibatch=args.batch_size, is_cuda=args.is_cuda,
#                         is_train=False)
#    model_dict = torch.load(args.models_root + "DHN.pth")
#    deepMunkres.load_state_dict(model_dict)
    
    print("loading appearance model from: ")
    print(  'pretrained/DAN.pth')
    sst = build_sst('test', 900)
    sst.load_state_dict(torch.load( 'pretrained/DAN.pth'))

    if args.is_cuda:
        sot_tracker.cuda()
        sst.cuda()
    
    print('Initializing the robust OT matching...')
    
    if args.method == 'sinkhorn_naive':
        deepMunkres = Sinkhorn_custom(method='naive', epsilon=args.epsilon, max_iter = args.max_inner_iter)
    elif args.method == 'sinkhorn_stablized':
        deepMunkres = Sinkhorn_custom(method='stablized', epsilon=args.epsilon, max_iter = args.max_inner_iter)
    elif args.method == 'sinkhorn_manual':
        deepMunkres = Sinkhorn_custom(method='manual', epsilon=args.epsilon, max_iter = args.max_inner_iter)
    elif args.method == 'sinkhorn_robust':
        deepMunkres = Sinkhorn_custom(method='robust', epsilon=args.epsilon, max_iter = args.max_inner_iter, \
                                rho1=args.rho1, rho2=args.rho2, eta=args.eta )

    # use gpu #
    if args.is_cuda:
#        deepMunkres = deepMunkres.cuda()
        sot_tracker.cuda()
        sst.cuda()
    sst.eval()

    # TensorboardX logs #
    print("creating logs files...")
    print("log path: ", args.logs + 'train_log')
    if os.path.exists(args.logs + 'train_log'):
        shutil.rmtree(args.logs + 'train_log')
    loss_writer = SummaryWriter(args.logs + 'train_log/loss')

    main(args, sot_tracker, deepMunkres, sst, optimizer, loss_writer)
