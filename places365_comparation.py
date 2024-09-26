import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Define the models
models_dict = {
    'alexnet': models.alexnet(pretrained=True),
    'resnet50': models.resnet50(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    # Assuming geffnet is a custom model, replace with actual import and initialization
    'geffnet': None  # Replace with actual geffnet model initialization
}

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load an example image
url = "https://example.com/path/to/your/image.jpg"  # Replace with actual image URL
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Preprocess the image
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# Function to get the top 5 predictions
def get_top5_predictions(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, indices = torch.topk(output, 5)
    return indices[0].tolist()

# Compare results
results = {}
for model_name, model in models_dict.items():
    if model is not None:
        top5 = get_top5_predictions(model, batch_t)
        results[model_name] = top5

# Print results
for model_name, top5 in results.items():
    print(f"Top 5 predictions for {model_name}: {top5}")