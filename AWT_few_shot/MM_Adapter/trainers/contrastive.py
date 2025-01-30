import torch
import torch.nn as nn

class Proximity(nn.Module):

    def __init__(self, num_classes, feat_dim, dtype, use_gpu):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=self.dtype).to(self.device))
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist_ = distmat * mask.float()
        count = torch.count_nonzero(dist_)   #0.0以外の値の数をカウント（.mean()だと0.0が含まれている要素も含めてしまう）
        loss_ = dist_.clamp(min=1e-12, max=1e+12).sum() / count
        return loss_

class Con_Proximity(nn.Module):

    def __init__(self, num_classes, feat_dim, dtype, use_gpu):
        super(Con_Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=self.dtype).to(self.device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        k_= mask.clone().to(dtype=torch.int8)

        k_= -1* k_ +1

        kk_= k_.clone().to(dtype=torch.bool)#uint8

        dist_ = distmat * kk_.float()
        count = torch.count_nonzero(dist_)   #0.0以外の値の数をカウント（.mean()だと0.0が含まれている要素も含めてしまう）

        loss_ = dist_.clamp(min=1e-12, max=1e+12).sum() / count

        return loss_