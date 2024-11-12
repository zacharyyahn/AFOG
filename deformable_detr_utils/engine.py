# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

from skimage.metrics import structural_similarity as ssim
import numpy as np


import torch
import time
import deformable_detr_utils.misc as utils
from dataset_utils.coco.coco_eval import CocoEvaluator
from dataset_utils.coco.panoptic_eval import PanopticEvaluator
from dataset_utils.coco.data_prefetcher import data_prefetcher


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, attack, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    total_time = 0.0
    total_l2 = 0.0
    total_l0 = 0.0
    total_mean_pert = 0.0
    total_median_pert = 0.0
    total_ssim = 0.0
    max_diff = 0.0
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        i += 1
        if float(args.load_attack) == 1.0:
            import numpy as np
            im_id = str(targets[0]["image_id"][0].item())
            im_name = "0" * (12 - len(im_id)) + im_id + ".npz"
            arr = np.load(args.load_dir + im_name)["arr_0"]
            arr = torch.tensor(np.resize(arr, samples.tensors[0].shape)) / 255.0
            samples.tensors[0] += arr
        orig_image = samples.tensors[0].clone()

        if attack != None:
            mult = 1.0 / torch.max(orig_image).item()
            start_time = time.perf_counter()
            samples = attack(model, samples.tensors, mode=args.attack_mode)
            end_time = time.perf_counter()
            
            # Calculate the metrics
            max_diff = max(torch.max(torch.abs(mult * orig_image - mult * samples)), max_diff)
            total_mean_pert += torch.mean(torch.abs(mult * orig_image - mult * samples))
            total_median_pert += torch.median(torch.abs(mult * orig_image - mult * samples))
            np_orig = mult * orig_image.clone().detach().cpu().numpy()
            np_sample = mult * samples.clone().squeeze().detach().cpu().numpy()
            total_ssim += ssim(np_orig, np_sample, data_range=(1.0), channel_axis=0)
            total_l2 += torch.linalg.vector_norm(mult * orig_image - mult * samples, 2) / (10e-3 * orig_image.shape[0] * orig_image.shape[1] * orig_image.shape[2])
            total_l0 += torch.count_nonzero(orig_image - samples) / (float(orig_image.shape[0]) * orig_image.shape[1] * orig_image.shape[2])
            total_time += (end_time - start_time)
            samples = samples.float()
        
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    print("\n\n------- Similarity and Timing Metrics -------")
    print(f"Average Attack Time: %0.4f" % (total_time / i))
    print(f"Average L2 Norm: %0.4f" % (total_l2 / i))
    print(f"Average L0 Norm: %0.4f" % (total_l0 / i))
    print(f"Average SSIM %0.4f" % (total_ssim / i))
    print(f"Mean Difference in Images: %0.4f" % (total_mean_pert / i))
    print(f"Median Difference in Images: %0.4f" % (total_median_pert / i))
    print(f"Max Difference in Images: %0.4f" % max_diff)
    print("------------------------------------------------\n\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
