from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data

# the outputs includes: 'boxes', 'labels', 'masks', 'scores'


def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    n = len(output_list)
    ap = {}
    for i in range(1, num_classes):
        ap[i] = 0
        gt_sum = 0
        score = []
        pred_bbox = []
        gt_bbox = []
        for j in range(n):
            if int(gt_labels_list[j]['labels']) == i:
                gt_sum += 1
            for k in range(output_list[j]['scores'].shape[0]):
                if int(output_list[j]['labels'][k]) == i:
                    score.append(float(output_list[j]['scores'][k]))
                    pred_bbox.append(output_list[j]['boxes'][k])
                    gt_bbox.append(j)
        dec_idx = np.argsort(-np.array(score))
        vis = {}  # 为了避免重复检测，记录vis字典
        tp = 0
        cnt = 0
        ap_data = []
        ap_data.append((0, 1))
        for j in dec_idx:
            cnt += 1
            if int(gt_labels_list[gt_bbox[j]]
                   ['labels']) == i and gt_bbox[j] not in vis.keys():
                iou = 0  # 用bbox计算iou
                bbox1 = pred_bbox[j]
                bbox2 = gt_labels_list[gt_bbox[j]]['boxes'][0]
                min_x = np.max([bbox1[0], bbox2[0]])
                max_x = np.min([[bbox1[2], bbox2[2]]])
                min_y = np.max([[bbox1[1], bbox2[1]]])
                max_y = np.min([[bbox1[3], bbox2[3]]])
                if min_x > max_x or min_y > max_y:
                    iou = 0
                else:
                    intersection = (max_x - min_x) * (max_y - min_y)
                    union = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0]) + (
                        bbox2[3] - bbox2[1]) * (bbox2[2] -
                                                bbox2[0]) - intersection
                    iou = intersection / union
                if iou >= iou_threshold:
                    tp += 1
                    vis[gt_bbox[j]] = True
            precision = tp / cnt
            recall = tp / gt_sum
            ap_data.append((recall, precision))
        ap_data.append((1, ap_data[-1][1]))
        for j in range(1, len(ap_data)):
            pre_recall, pre_precision = ap_data[j - 1]
            now_recall, now_precision = ap_data[j]
            ap[i] += (now_recall - pre_recall) * (
                now_precision + pre_precision
            ) / 2  # 用梯形近似面积，这样对于重复的recall值，pre_precision取得pre_recall下的最小值，而now_precision取得now_recall下的最大值
    mAP_detection = (ap[1] + ap[2] + ap[3]) / 3
    print(mAP_detection)
    return mAP_detection


def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    n = len(output_list)
    ap = {}
    for i in range(1, num_classes):
        ap[i] = 0
        gt_sum = 0
        score = []
        pred_mask = []
        gt_mask = []
        for j in range(n):
            if int(gt_labels_list[j]['labels']) == i:
                gt_sum += 1
            for k in range(output_list[j]['scores'].shape[0]):
                if int(output_list[j]['labels'][k]) == i:
                    score.append(float(output_list[j]['scores'][k]))
                    pred_mask.append(output_list[j]['masks'][k])
                    gt_mask.append(j)
        dec_idx = np.argsort(-np.array(score))
        vis = {}  # 为了避免重复检测，记录vis字典
        tp = 0
        cnt = 0
        ap_data = []
        ap_data.append((0, 1))
        for j in dec_idx:
            cnt += 1
            if int(gt_labels_list[gt_mask[j]]
                   ['labels']) == i and gt_mask[j] not in vis.keys():
                iou = 0  # 用mask计算iou
                mask1 = pred_mask[j][0].numpy()
                mask1 = np.where(mask1 >= 0.5, 1, 0)
                mask2 = gt_labels_list[gt_mask[j]]['masks'][0].numpy()
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                iou = intersection / union
                if iou >= iou_threshold:
                    tp += 1
                    vis[gt_mask[j]] = True
            precision = tp / cnt
            recall = tp / gt_sum
            ap_data.append((recall, precision))
        ap_data.append((1, ap_data[-1][1]))
        for j in range(1, len(ap_data)):
            pre_recall, pre_precision = ap_data[j - 1]
            now_recall, now_precision = ap_data[j]
            ap[i] += (now_recall - pre_recall) * (
                now_precision + pre_precision
            ) / 2  # 用梯形近似面积，这样对于重复的recall值，pre_precision取得pre_recall下的最小值，而now_precision取得now_recall下的最大值
    mAP_segmentation = (ap[1] + ap[2] + ap[3]) / 3
    print(mAP_segmentation)
    return mAP_segmentation


dataset_test = SingleShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=utils.collate_fn)

num_classes = 4

# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')

# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(
    torch.load(
        r'D:/课程/计算机视觉导论/hw/04_assignment/04_assignment/MaskRCNN/results/maskrcnn_7.pth',
        map_location='cpu'))

model.eval()
path = "results/"
# save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path + str(i) + "_result.png", imgs, output[0])

# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(10):
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)

np.savetxt(path + "mAP.txt", np.asarray([mAP_detection, mAP_segmentation]))