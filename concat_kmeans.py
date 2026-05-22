import torch
import faiss
import warnings
import numpy as np
import torch.nn.functional as F
from eval_utils import cluster_metric
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.simplefilter("ignore")

def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, cluster_num, gpu=0, spherical=True, niter=300, nredo=20, seed=0)
    X = X.astype(np.float32)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I


if __name__ == "__main__":

    backbone = "ViT-B32"
    dataset = "ImageNet-Dogs"
    # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs",
    # "DTD", "UCF101", "ImageNet"
    # 'Cars', 'Pets', 'Flowers', 'Food', 'Aircraft'
    # ]
    tau = 0.007
    # [0.009, 0.008, 0.01, 0.006, 0.007,
    # 0.01, "0.01", 0.006
    # '0.01', '0.01', '0.009', '0.01', '0.008'
    # ]


    # backbone = "ViT-B16"
    # dataset = "CIFAR-10"
    # # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-Dogs", "DTD"]
    # tau = 0.01
    # # [0.01, 0.01, 0.01, 0.009, 0.01] ViT-B16
    # # [0.01, 0.012, 0.03, 0.009, 0.01] ViT-L14
    # # [0.007, 0.009, 0.03, 0.009, 0.01] RN50

    if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
        cluster_num = 10
    elif dataset == "CIFAR-20":
        cluster_num = 20
    elif dataset == "ImageNet-Dogs":
        cluster_num = 15
    elif dataset == "DTD":
        cluster_num = 47
    elif dataset == "UCF101":
        cluster_num = 101
    elif dataset == "ImageNet":
        cluster_num = 1000
    elif dataset == "Cars":
        cluster_num = 196
    elif dataset == "Pets":
        cluster_num = 37
    elif dataset == "Flowers":
        cluster_num = 102
    elif dataset == "Food":
        cluster_num = 101
    elif dataset == "Aircraft":
        cluster_num = 100

    else:
        raise NotImplementedError

    images_embedding = np.load("./data/" + backbone + "/" + dataset + "/" + dataset + "_image_embedding_test.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )
    labels = np.loadtxt("./data/" + backbone + "/" + dataset + "/" + dataset + "_labels_test.txt")

    nouns_embedding = np.load("./data/" + backbone + "/" + dataset + "/" + dataset + "_filtered_nouns_embedding.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda()
    nouns_num = nouns_embedding.shape[0]
    images_embedding = torch.from_numpy(images_embedding).cuda()
    image_num = images_embedding.shape[0]

    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
            images_batch = images_embedding[start:end]
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{image_num}]")
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    images_embedding = images_embedding.cpu().numpy()

    concat_embedding = retrieval_embedding

    concat_embedding = np.concatenate([images_embedding, retrieval_embedding], axis=1)

    preds = kmeans(concat_embedding, cluster_num)
    cluster_metric(labels, preds) 





