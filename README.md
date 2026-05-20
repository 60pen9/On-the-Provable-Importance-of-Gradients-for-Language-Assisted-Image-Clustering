# On-the-Provable-Importance-of-Gradients-for-Language-Assisted-Image-Clustering

This is the code for the paper "On-the-Provable-Importance-of-Gradients-for-Language-Assisted-Image-Clustering (ICCV 2025)".

Abstract: *This paper investigates the recently emerged problem of Language-assisted Image Clustering (LaIC), where textual semantics are leveraged to improve the discriminability of visual representations to facilitate image clustering. 
Due to the unavailability of true class names, one of core challenges of LaIC lies in how to filter positive nouns, i.e., those semantically close to the images of interest, from unlabeled wild corpus data.
Existing filtering strategies are predominantly based on the off-the-shelf feature space learned by CLIP; however, despite being intuitive, these strategies lack a rigorous theoretical foundation. 
To fill this gap, we propose a novel gradient-based framework, termed as GradNorm, which is theoretically guaranteed and shows strong empirical performance. In particular, we measure the positiveness of each noun based on the magnitude of gradients back-propagated from the cross-entropy between the predicted target distribution and the softmax output. Theoretically, we provide a rigorous error bound to quantify the separability of positive nouns by GradNorm and prove that GradNorm naturally subsumes existing filtering strategies as extremely special cases of itself. Empirically, extensive experiments show that GradNorm achieves the state-of-the-art clustering performance on various benchmarks.*

# Usage

To improve the readability and extendibility of the code, we split different steps of our GradNorm method into separate `.py` files. Below is the step-by-step tutorial. Note that the intermediate results would be saved to the `./data` folder.

## Image and Text Embedding Inference
We first need to compute the image embedding with the CLIP model by running

> python image_embedding.py

and the embedding of WordNet nouns (provided in the `./data` folder) for text space construction by running

> python text_embedding.py

## Text Counterpart Construction
Next, we aim to find discriminative nouns to describe image semantic centers. Motivated by the zero-shot classification paradigm of CLIP, we reversely classify all nouns into $k$ image semantic centers and select the top confident nouns for each image semantic center by running

> python filter_nouns.py

The selected nouns compose the text space catering to the input images. Then, we retrieve nouns for each image to compute its counterpart in the text modality by running

> python retrieve_text.py

## Training-free Clustering
After the text counterpart construction, we arrive at an extremely simple baseline by applying $k$-means on the concatenated image and text features by running

> python concat_kmeans.py

Notably, such an implementation requires no additional training or modifications on CLIP, but it could significantly improve the clustering performance compared with directly applying $k$-means on the image embeddings.


Our codebase accesses the datasets from `./data/`by default.
```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── ...
```
