# On the Provable Importance of Gradients for Language Assisted Image Clustering

This is the code for the paper "On the Provable Importance of Gradients for Language Assisted Image Clustering (ICCV 2025)".

Abstract: *This paper investigates the recently emerged problem of Language-assisted Image Clustering (LaIC), where textual semantics are leveraged to improve the discriminability of visual representations to facilitate image clustering. 
Due to the unavailability of true class names, one of core challenges of LaIC lies in how to filter positive nouns, i.e., those semantically close to the images of interest, from unlabeled wild corpus data.
Existing filtering strategies are predominantly based on the off-the-shelf feature space learned by CLIP; however, despite being intuitive, these strategies lack a rigorous theoretical foundation. 
To fill this gap, we propose a novel gradient-based framework, termed as GradNorm, which is theoretically guaranteed and shows strong empirical performance. In particular, we measure the positiveness of each noun based on the magnitude of gradients back-propagated from the cross-entropy between the predicted target distribution and the softmax output. Theoretically, we provide a rigorous error bound to quantify the separability of positive nouns by GradNorm and prove that GradNorm naturally subsumes existing filtering strategies as extremely special cases of itself. Empirically, extensive experiments show that GradNorm achieves the state-of-the-art clustering performance on various benchmarks.*

# Usage
Before we formally start, donwload [pre-trained weights](https://1drv.ms/f/c/9ec3074c407dadd3/IgCsXBnSD0AeQ6Je0C7Bf5JLARi6MH31zwV0yEwh4RmnB2g?e=SXvuol) and move it to the `./data` folder.

## Image and Text Embedding Inference
We first need to compute the image embedding with the CLIP model by running

> python image_embedding.py

and the embedding of WordNet nouns (provided in the `./data` folder) for text space construction by running

> python text_embedding.py

## Text Counterpart Construction
Next, we aim to find discriminative nouns to describe images of interest by running

> python filter_nouns.py

## Training-free Clustering
After the text counterpart construction, we arrive at an extremely simple baseline by applying $k$-means on the concatenated image and text features by running

> python concat_kmeans.py


