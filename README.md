# Image Classification with ResNet50 (Pretrained) - 5 Scene Classes

This project implements an image classification model using a pretrained ResNet50 architecture. The model is trained on a dataset containing 5 scene classes:

- `airport_terminal`
- `aquarium`
- `beach`
- `bar`
- `music_studio`

## Model Architecture

We use the ResNet50 model, pretrained on the ImageNet dataset, as the base feature extractor. The parameters in the feature extractor layers are frozen to avoid updating the weights during training. A new fully connected layer (classification head) is added and fine-tuned to classify the images into the 5 scene classes.

### Model Configuration:
- **Pretrained model**: ResNet50 (pretrained on ImageNet)
- **New fully connected layers**:
    - Flatten
    - Linear(2048, 512) → ReLU
    - Dropout(0.2)
    - Linear(512, 128) → ReLU
    - Linear(128, 5)
- **Loss function**: Cross Entropy Loss
- **Optimizer**: Adam (with learning rate `1e-6`), optimizing only the new fully connected layers
