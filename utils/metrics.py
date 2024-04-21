import numbers
from typing import Optional
import torch


class ConfusionMatrix():
    _state_dict_all_req_keys = ("confusion_matrix", "_num_examples")
    def __init__(self,num_classes: int, device='cuda', average: Optional[str] = None,):
        self.num_classes = num_classes
        self._num_examples = 0
        self.device = device
        self.average = average

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self.device)
        self._num_examples = 0

    def pred_one_hot(self,pred):
        seg_map = torch.zeros([pred.size(0), pred.size(2), pred.size(3), pred.size(1)]).cuda()
        pred = torch.argmax(pred, dim=1)
        for k in range(pred.size(0)):
            seg_map[k] = torch.nn.functional.one_hot(pred[k], num_classes=21)

        seg_map = torch.permute(seg_map, (0, 3, 1, 2))

        return seg_map

    def update(self, pred, gt):
        y_pred = self.pred_one_hot(pred)

        self._num_examples += y_pred.shape[0]

        y_pred = torch.argmax(y_pred,dim=1).flatten()
        # y = torch.argmax(gt, dim=1).flatten()
        y = gt.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

def IoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None):
    if not isinstance(cm, ConfusionMatrix):
        raise TypeError(f"Argument cm should be instance of ConfusionMatrix, but given {type(cm)}")

    if not (cm.average in (None, "samples")):
        raise ValueError("ConfusionMatrix should have average attribute either None or 'samples'")

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError(
                f"ignore_index should be integer and in the range of [0, {cm.num_classes}), but given {ignore_index}"
            )
    cm = cm.confusion_matrix

    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:
        ignore_idx: int = ignore_index  # used due to typing issues with mympy

        def ignore_index_fn(iou_vector: torch.Tensor) -> torch.Tensor:
            if ignore_idx >= len(iou_vector):
                raise ValueError(f"ignore_index {ignore_idx} is larger than the length of IoU vector {len(iou_vector)}")
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_idx)
            return iou_vector[indices]
        return iou
    else:
        return iou


def mIoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None):
    iou = IoU(cm=cm, ignore_index=ignore_index).mean()
    return iou