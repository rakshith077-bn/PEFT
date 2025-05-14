# An image is worth 16x16 words - Summary 

## Introduction
When pre-trainned on large amounts of data and transfered to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc), ViT attains excellent results compared to the state-of the art CNNs while requiring substantially fewer computational resources to train.

- Principle: Apply standard Transformer directly to images, with the fewest possible modifications. To do so, the image is split into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as 'Tokens' in an NLP application. The model is trainned on image classification in supervised fashion.

- Drawback - Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trainned on insufficient data. Generally the a good result is expected if the model is trainned on atleast 14M-300M images. 
