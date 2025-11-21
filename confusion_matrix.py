import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import BrainTumorDataset
from models import create_mobilenetv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = BrainTumorDataset("data/Testing", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.class_names

# Load saved model
model = create_mobilenetv2(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load("mobilenet_brain_tumor.pth", map_location=DEVICE))
model.eval()

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute and Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Confusion Matrix - Mobilenet Brain Tumor Classification")
plt.show()
