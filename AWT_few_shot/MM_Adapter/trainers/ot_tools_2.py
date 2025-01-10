import torch
import torch.nn.functional as F


def pairwise_distances(image_features, text_features):
    # image_features: aug_time x d, text_features: n_des x d
    # aug_time x d - n_des x d -> aug_time x n_des
    image_features = image_features.to(torch.float32)
    text_features = text_features.to(torch.float32)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    dist = torch.cdist(image_features, text_features, p=2)
    return dist


def Sinkhorn(K, u, v, thresh=1e-2, max_iter=200):
    # K: aug_time x n_des
    r = torch.ones_like(u) # aug_time x 512
    c = torch.ones_like(v) # n_des x 512
    for i in range(max_iter):
        r0 = r
        r = u / torch.matmul(K, c) # size: aug_time x 512 
        c = v / torch.matmul(K.t(), r) # size: n_des x 512
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = (r.unsqueeze(1) * K) * c.unsqueeze(0)
    return T


def optimal_transport(image_features, text_features, eps=1, thresh=1e-2, max_iter=200):
    # image_features: aug_time x d, text_features: n_des x d
    aug_time = image_features.shape[0]
    n_des = text_features.shape[0]

    C = pairwise_distances(image_features, text_features) # aug_time x n_des
    
    a = torch.full((aug_time,), 1.0/aug_time, device=image_features.device)
    b = torch.full((n_des,), 1.0/n_des, device=text_features.device)
    
    with torch.no_grad():
        K = torch.exp(-C / eps) # K: aug_time x n_des
        u = torch.ones_like(a) # aug_time x 512
        v = torch.ones_like(b) # n_des x 512

        for i in range(max_iter):
            u0 = u
            u = a / torch.matmul(K, v) # size: aug_time x 512 
            v = b / torch.matmul(K.t(), u) # size: n_des x 512
            
            err = (u - u0).abs().mean()
            if err.item() < thresh:
                break

        T = (u.unsqueeze(1) * K) * v.unsqueeze(0)

    assert not torch.isnan(T).any()

    wass_dist = torch.sum(T * C) # aug_time x n_des
    wass_dist = wass_dist
    return wass_dist


def Wasserstein_Distance(image_features, text_features):
    # logits: bs x aug_time x n_des x n_cls
    # calculate wasserstein distance for every batch
    wass_dist = optimal_transport(image_features, text_features)

    return wass_dist

