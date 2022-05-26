# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def hc_loss(logits, labels, reduction="mean"):
    # 一个不会溢出的 trick
    log_pt = F.log_softmax(logits, dim=-1)  # 这里相当于 CE loss
    pts = torch.exp(log_pt)  # 通过 softmax 函数后打的分
    # print(f"pt: {pts}")
    _, top2 = pts.topk(k=2, dim=1)
    top1_preds = top2[:, 0]
    top2_preds = top2[:, 1]
    # print(f"top1_preds: {top1_preds}, top2_preds: {top2_preds}")
    hc_loss = 0
    for truth_label, top1_pred, top2_pred, pt in zip(labels, top1_preds, top2_preds, pts):
        # print(f"truth_label: {truth_label}")
        # print(f"top1_pred: {top1_pred}")
        # print(f"top2_pred: {top2_pred}")
        # print(f"pt: {pt}")
        # 如果预测和真值相同，则取top2
        if truth_label == top1_pred:
            # hc_loss += torch.sigmoid(pt[top2_pred])
            # hc_loss += pt[top2_pred]
            # hc_loss += torch.log(1 + pt[top2_pred])
            pass
        else:
            # 预测错误，取top1
            hc_loss += -torch.log(pt[top1_pred])
    
    hc_loss = 10 * hc_loss / len(labels)

    return hc_loss

    


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], 
                           [0.6, 0.4, 0.9, 0.5],
                           [0.1, 0.4, 0.2, 0.6],], requires_grad=True)
    labels = torch.tensor([1, 2, 2])
    print(hc_loss(logits, labels))