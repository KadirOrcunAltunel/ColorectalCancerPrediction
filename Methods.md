#### VISION TRANSFORMER

The core idea of ViT is to treat an image as a sequence of patches. An image is divided into fixed-size patches (16x16 pixels), and each patch is then flattened into a vector. These vectors are linearly embedded and combined with positional embeddings to retain spatial information. This sequence of embedded patches is then fed into the standard Transformer encoder, which consists of layers of multi-head self-attention and feed-forward neural networks (FFNs). It is pre trained on ImageNet-21k, which has 14 million images and 21843 classes at a resolution of 224x224.

![ViT](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/ViT.png)


The self-attention mechanism allows the model to weigh the importance of different patches relative to each other, enabling it to capture complex patterns and relationships across the entire image. Attention weights are interpreted as a probability distribution, which is how importance is chosen for the patches. The output from the Transformer encoder is then passed through a classification layer (linear) to produce the final class predictions
