import torch
import faiss
import numpy as np
import torch.nn.functional as F
from concat_kmeans import run

def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=0, spherical=True, niter=300, nredo=20, seed=0)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I


if __name__ == "__main__":

    backbone = "ViT-B32"
    dataset = "CIFAR-20"
    # 　["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs",
    # "DTD", "UCF101", "ImageNet",
    # 'ImageNet-C', 'ImageNet-V2', 'ImageNet-S', 'ImageNet-A','ImageNet-R'
    # 'Cars', 'Pets', 'Flowers', 'Food', 'Aircraft']
    cluster_num = 200
    # [250, 200, 30, 50, 50,
    # 180, 303, 2000
    # 'ImageNet-C', 'ImageNet-V2', 'ImageNet-S', 'ImageNet-A', 'ImageNet-R'
    # 'Cars', '120', '300', '1200', '300'
    # ]
    temp = 0.08
    #[0.06, 0.08, 0.08, 0.018, 0.02,
    # 0.08, 303, 0.02
    # 0.02, '0.02', '0.03', '0.008', '0.02'
    # '0.01', '0.01', '0.01', '0.01', '0.02'
    # ]
    topK = 5
    #[5, 5, 5, 5, 5,
    # 5, 303, 3
    # 'ImageNet-C', '1', '3', '1', '3'
    # '5', '5', '5', '5', '5'
    # ]
    p = 2.0


    # backbone = "ViT-L14"
    # dataset = "DTD"
    # # ["CIFAR-10", "CIFAR-20", "DTD"]
    # cluster_num = 50
    # # [250, 200, 30, 50, 180] "ViT-B16" "ViT-L16""RN50"
    # temp = 0.08
    # # [0.06, 0.08, 0.08, 0.02/, 0.08] "ViT-B16/ViT-L16"
    # # [0.1, 0.08, 0.08, 0.02, 0.1] "RN50"
    # topK = 5
    # #[5, 5, 5, 5, 5]
    # p = 2.0

    nouns_embedding = np.load("./data/" + backbone + "/nouns_embedding_ensemble.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    print(nouns_embedding.shape)

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]

    weights = np.load(
        "./data/" + backbone + "/" + dataset + "/" + dataset + "_weights_" + str(cluster_num) + "cluster.npy"
    )

    weights = torch.from_numpy(weights).cuda()

    similarity = torch.matmul(weights, nouns_embedding.T)/temp
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()
    nouns_embedding=nouns_embedding.cpu()
    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)


    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        k_nouns = nouns_embedding[class_index]
        softmax_class = softmax_nouns[:, class_index].T
        confidence = softmax_class.max(dim=1)[0]

        score_1 = torch.sum(torch.pow(softmax_class, p), dim=1)+torch.pow(1-confidence, p)-torch.pow(confidence, p)
        score_2 = torch.sum(torch.pow(torch.abs(k_nouns),p), dim=1)
        score = torch.pow(score_1*score_2, 1/p)

        rank = torch.argsort(-score, descending=True)
        selected_idx[class_index[rank[:topK]]] = True
    selected_idx = selected_idx.cpu().numpy()

    print(selected_idx.sum(), "nouns selected.")
    nouns_embedding_selected = nouns_embedding[selected_idx]
    np.save(
        "./data/" + backbone + "/" + dataset + "/" + dataset + "_filtered_nouns_embedding.npy",
        nouns_embedding_selected.cpu().numpy(),
    )





