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

def run(dataset, backbone):
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
    elif dataset == "ImageNet" or dataset == "ImageNet-C" or dataset == "ImageNet-V2" or dataset == "ImageNet-S":
        cluster_num = 1000
    elif dataset == "ImageNet-R" or dataset == "ImageNet-A":
        cluster_num = 200
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


    for tau in [
        # 0.0004, 0.0006, 0.0008, 0.001,
        0.002, 0.003, 0.004,
        0.005, 0.006, 0.007, 0.008, 0.009,
        0.01, 0.012, 0.014, 0.016, 0.018,
        # 0.02,0.03,0.04,0.05,
        # 0.06,0.07,0.08,0.09,
        # 0.1,0.2
    ]:
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

        concat_embedding = np.concatenate([images_embedding.cpu().numpy(), retrieval_embedding], axis=1)
        preds = kmeans(concat_embedding, cluster_num)
        print(tau)
        cluster_metric(labels, preds)




if __name__ == "__main__":

    backbone = "ViT-B32"
    dataset = "ImageNet-Dogs"
    # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs",
    # "DTD", "UCF101", "ImageNet"
    # 'ImageNet-C', 'ImageNet-V2', 'ImageNet-S', 'ImageNet-A','ImageNet-R',
    # 'Cars', 'Pets', 'Flowers', 'Food', 'Aircraft'
    # ]
    tau = 0.007
    # [0.009, 0.008, 0.01, 0.006, 0.007,
    # 0.01, "UCF101", 0.006
    # 0.007, 0.004, '0.006', '0.007', '0.005'
    # '0.01', '0.01', '0.009', '0.01', '0.008'
    # ]


    # backbone = "ViT-B16"
    # dataset = "CIFAR-10"
    # # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-Dogs", "DTD"]
    # tau = 0.01
    # [0.01, 0.01, 0.01, 0.009, 0.01] ViT-B16
    # [0.01, 0.012, 0.03, 0.009, 0.01] ViT-L14
    # [0.007, 0.009, 0.03, 0.009, 0.01] RN50

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
    elif dataset == "ImageNet" or dataset == "ImageNet-C" or dataset == "ImageNet-V2" or dataset == "ImageNet-S":
        cluster_num = 1000
    elif dataset == "ImageNet-R" or dataset == "ImageNet-A":
        cluster_num = 200
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

    # images_embedding = np.load("./data/" + dataset + "_image_embedding_train.npy")
    # images_embedding = images_embedding / np.linalg.norm(
    #     images_embedding, axis=1, keepdims=True
    # )
    # labels = np.loadtxt("./data/" + dataset + "_labels_train.txt")

    nouns_embedding = np.load("./data/" + backbone + "/" + dataset + "/" + dataset + "_filtered_nouns_embedding.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    # tsne = TSNE(n_components=2, random_state=42)
    # images_embedding_tsne = tsne.fit_transform(images_embedding)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(images_embedding_tsne[:, 0], images_embedding_tsne[:, 1], c=labels, cmap='tab20', marker='+', s=5)
    # # plt.colorbar()  # Show color bar for the target classes
    # ax = plt.gca()
    # plt.yticks(size=10)
    # plt.xticks(size=10)
    # ax.set_ylim([-100, 100])
    # ax.set_xlim([-100, 100])
    # # plt.xlabel("t-SNE component 1")
    # # plt.ylabel("t-SNE component 2")
    # # plt.show()
    # plt.savefig('Figure_1.png', dpi=1000, bbox_inches='tight')

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
    cluster_metric(labels, preds) #[91.1, 60.6, 98.3, 99.4, 81.2, 50.9, "UCF101", 49.9]

    # tsne = TSNE(n_components=2, random_state=42)
    # concat_embedding_tsne = tsne.fit_transform(retrieval_embedding)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(concat_embedding_tsne[:, 0], concat_embedding_tsne[:, 1], c=labels, cmap='tab20', marker='+', s=3)
    # ax = plt.gca()
    # plt.yticks(size=10)
    # plt.xticks(size=10)
    # ax.set_ylim([-100, 100])
    # ax.set_xlim([-100, 100])
    # # plt.xlabel("t-SNE component 1")
    # # plt.ylabel("t-SNE component 2")
    # # plt.show()
    # plt.savefig('Figure_2.png', dpi=1000, bbox_inches='tight')
    #
    # tsne = TSNE(n_components=2, random_state=42)
    # concat_embedding_tsne = tsne.fit_transform(concat_embedding)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(concat_embedding_tsne[:, 0], concat_embedding_tsne[:, 1], c=labels, cmap='tab20', marker='+', s=3)
    # ax = plt.gca()
    # plt.yticks(size=10)
    # plt.xticks(size=10)
    # ax.set_ylim([-100, 100])
    # ax.set_xlim([-100, 100])
    # # plt.xlabel("t-SNE component 1")
    # # plt.ylabel("t-SNE component 2")
    # # plt.show()
    # plt.savefig('Figure_3.png', dpi=1000, bbox_inches='tight')








    # images_embedding_train = np.load("./data/" + dataset + "_image_embedding_train.npy")
    # images_embedding_train = images_embedding_train / np.linalg.norm(
    #     images_embedding_train, axis=1, keepdims=True
    # )
    #
    # images_embedding_test = np.load("./data/" + dataset + "_image_embedding_test.npy")
    # images_embedding_test = images_embedding_test / np.linalg.norm(
    #     images_embedding_test, axis=1, keepdims=True
    # )
    # labels = np.loadtxt("./data/" + dataset + "_labels_test.txt")
    #
    # nouns_embedding = np.load("./data/" + dataset + "_filtered_nouns_embedding.npy")
    # nouns_embedding = nouns_embedding / np.linalg.norm(
    #     nouns_embedding, axis=1, keepdims=True
    # )
    #
    # nouns_embedding = torch.from_numpy(nouns_embedding).cuda()
    # nouns_num = nouns_embedding.shape[0]
    #
    # images_embedding_train = torch.from_numpy(images_embedding_train).cuda()
    # image_num_train = images_embedding_train.shape[0]
    #
    # images_embedding_test = torch.from_numpy(images_embedding_test).cuda()
    # image_num_test = images_embedding_test.shape[0]


    # tau = 0.009 # [0.009, 0.006, 0.01, 0.006, 0.007, 0.008, "UCF101", 0.006]
    # lamda = 0.009 # [0.009, 0.006, 0.01, 0.006, 0.007, 0.005, "UCF101", 0.006]
    # batch_size = 8192
    #
    # retrieval_similarity = []
    # for i in range(image_num_train // batch_size + 1):
    #     start = i * batch_size
    #     end = start + batch_size
    #     if end > image_num_train:
    #         end = image_num_train
    #         images_batch = images_embedding_train[start:end]
    #     similarity = torch.matmul(images_embedding_train[start:end], nouns_embedding.T)
    #     similarity = torch.softmax(similarity / tau, dim=1)
    #     retrieval_similarity.append(similarity)
    #     if i % 50 == 0:
    #         print(f"[Completed {i * batch_size}/{image_num_train}]")
    #
    # retrieval_similarity = torch.cat(retrieval_similarity, dim=0).cuda()
    # retrieval_similarity = torch.mean(retrieval_similarity, dim=0, keepdim=True)
    #
    # retrieval_embeddings = []
    # for i in range(image_num_test // batch_size + 1):
    #     start = i * batch_size
    #     end = start + batch_size
    #     if end > image_num_test:
    #         end = image_num_test
    #         images_batch = images_embedding_test[start:end]
    #     similarity = torch.matmul(images_embedding_test[start:end], nouns_embedding.T)
    #     similarity = similarity - lamda*torch.log(retrieval_similarity)
    #     similarity = torch.softmax(similarity / tau, dim=1)
    #     retrieval_embedding = (similarity @ nouns_embedding).cpu()
    #     retrieval_embeddings.append(retrieval_embedding)
    #     if i % 50 == 0:
    #         print(f"[Completed {i * batch_size}/{image_num_test}]")
    #
    # retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda()
    # retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    # images_embedding_test = images_embedding_test.cpu().numpy()
    #
    # concat_embedding = np.concatenate([images_embedding_test, retrieval_embedding], axis=1)
    # preds = kmeans(concat_embedding, cluster_num)
    # cluster_metric(labels, preds) #[91.1, 60.6, 98.3, 99.4, 81.2, 50.9, "UCF101", 49.9]






