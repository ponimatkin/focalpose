import torch
import torchvision
import numpy as np
from focalpose.lib3d.camera_geometry import get_K_crop_resize


def crop_to_aspect_ratio(images, box, masks=None, K=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (torch.arange(bsz).unsqueeze(1).to(box.device).float(), box.unsqueeze(0).repeat(bsz, 1).float()),
        dim=1).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(K, boxes[:, 1:], orig_size=(h, w), crop_resize=(h_output, w_output))
    return images, masks, K


def make_detections_from_segmentation(masks):
    detections = []
    if masks.dim() == 4:
        assert masks.shape[0] == 1
        masks = masks.squeeze(0)

    for mask_n in masks:
        dets_n = dict()
        for uniq in torch.unique(mask_n, sorted=True):
            ids = np.where((mask_n == uniq).cpu().numpy())
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            dets_n[int(uniq.item())] = torch.tensor([x1, y1, x2, y2]).to(mask_n.device)
        detections.append(dets_n)
    return detections


def make_masks_from_det(detections, h, w):
    n_ids = len(detections)
    detections = torch.as_tensor(detections)
    masks = torch.zeros((n_ids, h, w)).byte()
    for mask_n, det_n in zip(masks, detections):
        x1, y1, x2, y2 = det_n.cpu().int().tolist()
        mask_n[y1:y2, x1:x2] = True
    return masks

def perturb_detections(data):
    bboxes = data.bboxes
    bsz, _, h, w = data.images.shape
    dx_range = (5*(bboxes[:, 2] - bboxes[:, 0])//100).int()
    dy_range = (5*(bboxes[:, 3] - bboxes[:, 1])//100).int()

    dx1 = [torch.randint(-c.item(), c.item(), (1, )) if c != 0 else 0 for c in dx_range]
    dy1 = [torch.randint(-c.item(), c.item(), (1, )) if c != 0 else 0 for c in dy_range]
    dx2 = [torch.randint(-c.item(), c.item(), (1, )) if c != 0 else 0 for c in dx_range]
    dy2 = [torch.randint(-c.item(), c.item(), (1, )) if c != 0 else 0 for c in dy_range]
    perturbed_boxes = (bboxes - torch.tensor([dx1, dy1, dx2, dy2]).T).float()
    # This makes sure that bounding box is not outside image region
    perturbed_boxes = torch.stack([
        torch.max(torch.zeros(bsz), perturbed_boxes[:, 0]),
        torch.max(torch.zeros(bsz), perturbed_boxes[:, 1]),
        torch.min(w*torch.ones(bsz), perturbed_boxes[:, 2]),
        torch.min(h*torch.ones(bsz), perturbed_boxes[:, 3]),
    ]).T

    return perturbed_boxes
