"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# Try to import PyTorch and related modules; allow the script to be imported
# Require Python 3.10+ because some third-party libs (dinov2) use
# PEP 604 union types (e.g. `float | None`). Those raise a TypeError
# on older Python versions. Fail early with a helpful message.
if sys.version_info < (3, 10):
    sys.stderr.write(
        f"train_segmentation.py requires Python 3.10 or newer. "
        f"Current interpreter: {sys.version.split()[0]}\n"
    )
    sys.exit(1)

# Try to import PyTorch and related modules; allow the script to be imported
# in environments without torch for dry-run / path-checking purposes.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch import nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    # Provide lightweight fallbacks so import-time doesn't fail.
    Dataset = object
    DataLoader = None
    nn = None
    F = None
    optim = None
    transforms = None
    torchvision = None
    # tqdm fallback: identity iterator
    def tqdm(x, **kwargs):
        return x

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])
    
def calculate_iou(pred, target, num_classes):
    ious = []
    
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious

def calculate_miou(pred, target, num_classes):
    ious = calculate_iou(pred, target, num_classes)
    return np.nanmean(ious)


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)

    if arr.ndim > 2:
        arr = arr.squeeze()

    new_arr = np.zeros_like(arr, dtype=np.uint8)

    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value

    return torch.from_numpy(new_arr).long()
# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        # Both color images and masks are .png files with same name
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale to ensure consistent mode

        if self.transform:
            image = self.transform(image)

        # Apply any mask transforms (e.g., resize, to-tensor) before converting labels
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert raw mask values to class ids
        mask = convert_mask(mask)

        # Ensure tensor type/shape for training
        if TORCH_AVAILABLE:
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 3 and mask.size(0) == 1:
                    mask = mask.squeeze(0)
                mask = mask.long()
        else:
            # keep as numpy array for dry-run
            if not isinstance(mask, np.ndarray):
                try:
                    mask = np.array(mask)
                except Exception:
                    pass

        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================
if TORCH_AVAILABLE:

    class SegmentationHeadConvNeXt(nn.Module):
        def __init__(self, in_channels, out_channels, tokenH, tokenW):
            super().__init__()

            self.H = tokenH
            self.W = tokenW

            # Decoder block
            self.decoder = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),

                nn.Dropout2d(0.3)
            )

            # Final classifier
            self.classifier = nn.Conv2d(256, out_channels, kernel_size=1)

        def forward(self, x):
            """
            x shape from DINO backbone: (B, N, C)
            """

            B, N, C = x.shape

            # Reshape tokens to spatial map
            x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

            x = self.decoder(x)
            x = self.classifier(x)

            # Upsample back to original resolution
            x = F.interpolate(
                x,
                scale_factor=14,  # patch size of DINOv2
                mode="bilinear",
                align_corners=False
            )

            return x


    # ============================================================================
    # Loss Function: Focal Dice Loss
    # ============================================================================

    class FocalDiceLoss(nn.Module):
        """
        Combines Focal Loss and Dice Loss for better handling of class imbalance.
        """
        def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, smooth=1e-6):
            super().__init__()
            self.alpha = alpha  # weight for focal loss
            self.beta = beta    # weight for dice loss
            self.gamma = gamma  # focusing parameter for focal loss
            self.smooth = smooth

        def forward(self, pred, target):
            """
            pred: logits (B, C, H, W)
            target: ground truth (B, H, W)
            """
            # Softmax to get probabilities
            pred_probs = F.softmax(pred, dim=1)
            pred_log = F.log_softmax(pred, dim=1)

            # Reshape for easier computation
            B, C, H, W = pred.shape
            pred_probs = pred_probs.view(B, C, -1)
            pred_log = pred_log.view(B, C, -1)
            target = target.view(B, -1)

            # One-hot encode target
            target_one_hot = F.one_hot(target, num_classes=C).permute(0, 2, 1).float()

            # Focal Loss
            focal_loss = -(1 - pred_probs) ** self.gamma * target_one_hot * pred_log
            focal_loss = focal_loss.sum(dim=2).mean()

            # Dice Loss
            intersection = (pred_probs * target_one_hot).sum(dim=2)
            union = pred_probs.sum(dim=2) + target_one_hot.sum(dim=2)
            dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = dice_loss.mean()

            # Combined Loss
            total_loss = self.alpha * focal_loss + self.beta * dice_loss
            return total_loss


    # ============================================================================
    # Metrics
    # ============================================================================


    def compute_iou(pred, target, num_classes, ignore_index=255):
        """
        pred: logits (B, C, H, W)
        target: ground truth (B, H, W)
        """

        pred = torch.argmax(pred, dim=1)

        pred = pred.view(-1)
        target = target.view(-1)

        iou_per_class = []

        for cls in range(num_classes):
            if cls == ignore_index:
                continue

            pred_inds = pred == cls
            target_inds = target == cls

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append((intersection / union).cpu().item())

        return np.nanmean(iou_per_class)


    # ==========================================
    # Dice Score
    # ==========================================


    def compute_dice(pred, target, num_classes, smooth=1e-6):
        """
        pred: logits (B, C, H, W)
        target: ground truth (B, H, W)
        """

        pred = torch.argmax(pred, dim=1)

        dice_scores = []

        for cls in range(num_classes):
            pred_inds = (pred == cls).float()
            target_inds = (target == cls).float()

            intersection = (pred_inds * target_inds).sum()
            dice = (2. * intersection + smooth) / (
                pred_inds.sum() + target_inds.sum() + smooth
            )

            dice_scores.append(dice.cpu().item())

        return np.mean(dice_scores)


    # ==========================================
    # Pixel Accuracy
    # ==========================================


    def compute_pixel_accuracy(pred, target):
        """
        pred: logits (B, C, H, W)
        target: ground truth (B, H, W)
        """

        pred = torch.argmax(pred, dim=1)

        correct = (pred == target).float().sum()
        total = torch.numel(target)

        return (correct / total).cpu().item()


    # ==========================================
    # Full Evaluation Function
    # ==========================================


    def evaluate_metrics(model, backbone, dataloader, device, num_classes=10):
        """
        Runs full evaluation:
        Returns (mIoU, Dice, Pixel Accuracy)
        """

        model.eval()
        backbone.eval()

        iou_scores = []
        dice_scores = []
        pixel_accuracies = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Evaluating"):

                imgs = imgs.to(device)
                labels = labels.to(device).long()

                # Extract DINO features
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]

                # Get logits from head
                logits = model(features)

                # Compute metrics
                iou = compute_iou(logits, labels, num_classes)
                dice = compute_dice(logits, labels, num_classes)
                pixel_acc = compute_pixel_accuracy(logits, labels)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(pixel_acc)

        return (
            np.mean(iou_scores),
            np.mean(dice_scores),
            np.mean(pixel_accuracies)
        )

else:
    # Torch not available: define stubs that raise informative errors when used.
    class SegmentationHeadConvNeXt:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch is not available in this environment')

    class FocalDiceLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch is not available in this environment')

    def compute_iou(*args, **kwargs):
        raise RuntimeError('PyTorch is not available in this environment')

    def compute_dice(*args, **kwargs):
        raise RuntimeError('PyTorch is not available in this environment')

    def compute_pixel_accuracy(*args, **kwargs):
        raise RuntimeError('PyTorch is not available in this environment')

    def evaluate_metrics(*args, **kwargs):
        raise RuntimeError('PyTorch is not available in this environment')
# ============================================================================
# Plotting Functions
# ============================================================================



def save_training_plots(history, output_dir):
    """
    Saves all training curves:
    - Loss
    - IoU
    - Dice
    - Pixel Accuracy
    - Combined dashboard
    """

    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ================================
    # 1️⃣ LOSS CURVES
    # ================================
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)

    best_epoch = np.argmin(history["val_loss"]) + 1
    plt.axvline(best_epoch, linestyle="--", alpha=0.6,
                label=f"Best Val Epoch {best_epoch}")

    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=300)
    plt.close()

    # ================================
    # 2️⃣ IoU CURVES
    # ================================
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_iou"], label="Train IoU", linewidth=2)
    plt.plot(epochs, history["val_iou"], label="Val IoU", linewidth=2)

    best_epoch = np.argmax(history["val_iou"]) + 1
    plt.axvline(best_epoch, linestyle="--", alpha=0.6,
                label=f"Best Val Epoch {best_epoch}")

    plt.title("IoU vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_curves.png"), dpi=300)
    plt.close()

    # ================================
    # 3️⃣ DICE CURVES
    # ================================
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_dice"], label="Train Dice", linewidth=2)
    plt.plot(epochs, history["val_dice"], label="Val Dice", linewidth=2)

    best_epoch = np.argmax(history["val_dice"]) + 1
    plt.axvline(best_epoch, linestyle="--", alpha=0.6,
                label=f"Best Val Epoch {best_epoch}")

    plt.title("Dice Score vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_curves.png"), dpi=300)
    plt.close()

    # ================================
    # 4️⃣ PIXEL ACCURACY CURVES
    # ================================
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_pixel_acc"], label="Train Pixel Acc", linewidth=2)
    plt.plot(epochs, history["val_pixel_acc"], label="Val Pixel Acc", linewidth=2)

    best_epoch = np.argmax(history["val_pixel_acc"]) + 1
    plt.axvline(best_epoch, linestyle="--", alpha=0.6,
                label=f"Best Val Epoch {best_epoch}")

    plt.title("Pixel Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pixel_accuracy_curves.png"), dpi=300)
    plt.close()

    # ================================
    # 5️⃣ COMBINED DASHBOARD
    # ================================
    plt.figure(figsize=(14, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    # IoU
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_iou"], label="Train")
    plt.plot(epochs, history["val_iou"], label="Val")
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    # Dice
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_dice"], label="Train")
    plt.plot(epochs, history["val_dice"], label="Val")
    plt.title("Dice")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    # Pixel Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_pixel_acc"], label="Train")
    plt.plot(epochs, history["val_pixel_acc"], label="Val")
    plt.title("Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics_dashboard.png"), dpi=300)
    plt.close()

    print(f"All training plots saved to: {output_dir}")

# ============================================================================
# Main Training Function
# ============================================================================

def main():

    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import argparse

    # =====================================================
    # DEVICE
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =====================================================
    # HYPERPARAMETERS
    # =====================================================
    batch_size = 8
    lr = 1e-4
    n_epochs = 40
    n_classes = 10
    image_size = 768   # must be multiple of 14 for DINOv2
    freeze_backbone = True

    # Image size (multiple of 14 for DINOv2)
    w = (image_size // 14) * 14     
    h = (image_size // 14) * 14

    # =====================================================
    # TRANSFORMS
    # =====================================================
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.PILToTensor()
    ])

    # =====================================================
    # DATASET / ARGS
    # =====================================================
    parser = argparse.ArgumentParser(description="Train segmentation head")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to Offroad_Segmentation_Training_Dataset (contains train/ and val/)")
    parser.add_argument("--dry_run", action="store_true", help="Validate dataset paths and exit")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=n_epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--image_size", type=int, default=image_size,
                        help="Image size (must be multiple of 14 for DINOv2)")
    parser.add_argument("--smoke_test", action="store_true", help="Load one batch and run a forward pass then exit")
    args = parser.parse_args()

    if args.dataset_path:
        base_dataset = args.dataset_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dataset = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset")

    train_dir = os.path.join(base_dataset, "train")
    val_dir = os.path.join(base_dataset, "val")

    # Apply CLI overrides to hyperparameters before creating datasets/transforms
    batch_size = args.batch_size
    n_epochs = args.epochs
    lr = args.lr
    image_size = args.image_size

    # Recompute sizes and transforms based on potentially updated image_size
    w = (image_size // 14) * 14
    h = (image_size // 14) * 14

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.PILToTensor()
    ])

    # If dry run, report dataset paths and exit early to avoid FileNotFoundError
    if args.dry_run:
        def safe_print(label, value):
            try:
                print(label, value)
            except UnicodeEncodeError:
                # fallback to ascii-safe label and value using backslash escapes
                label_safe = label.encode('ascii', errors='backslashreplace').decode('ascii')
                value_safe = repr(value).encode('ascii', errors='backslashreplace').decode('ascii')
                print(label_safe, value_safe)

        safe_print('Dry run - Dataset root:', base_dataset)
        safe_print('  Train dir:', train_dir)
        safe_print('  Val dir:', val_dir)
        safe_print('  Train exists:', os.path.isdir(train_dir))
        safe_print('  Val exists:', os.path.isdir(val_dir))
        safe_print('  Train Color_Images exists:', os.path.isdir(os.path.join(train_dir, 'Color_Images')))
        safe_print('  Train Segmentation exists:', os.path.isdir(os.path.join(train_dir, 'Segmentation')))
        return

    # Validate dataset paths exist
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print('Error: Dataset directories not found.')
        print('  Tried train_dir:', train_dir)
        print('  Tried val_dir:', val_dir)
        print('\nUsage:')
        print('  python train_segmentation.py --dataset_path "C:\\path\\to\\Offroad_Segmentation_Training_Dataset"')
        print('  or')
        print('  python train_segmentation.py --dry_run --dataset_path "C:\\path\\to\\Offroad_Segmentation_Training_Dataset"')
        return

    trainset = MaskDataset(train_dir, transform, mask_transform)
    valset   = MaskDataset(val_dir, transform, mask_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(trainset)}")
    print(f"Val samples: {len(valset)}")

    # =====================================================
    # LOAD DINOv2 BACKBONE
    # =====================================================
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()
    backbone.to(device)#Freeze backbone initially
    if freeze_backbone:
        print("Freezing backbone parameters...")
        for param in backbone.parameters():
            param.requires_grad = False


    # Get embedding dimension
    sample_img, _ = next(iter(train_loader))
    sample_img = sample_img.to(device)

    with torch.no_grad():
        features = backbone.forward_features(sample_img)["x_norm_patchtokens"]
        embed_dim = features.shape[-1]

    print("Embedding dimension:", embed_dim)

    # =====================================================
    # SEGMENTATION HEAD
    # =====================================================
    tokenH = tokenW = image_size // 14
    classifier = SegmentationHeadConvNeXt(
        in_channels=embed_dim,
        out_channels=n_classes,
        tokenH=tokenH,
        tokenW=tokenW
    ).to(device)

    # =====================================================
    # LOSS + OPTIMIZER
    # =====================================================
    criterion = FocalDiceLoss(alpha=0.5, beta=0.5)
    optimizer = torch.optim.AdamW(list(classifier.parameters())+list(backbone.parameters()), lr=lr  ,weight_decay=1e-4)

    # If smoke_test requested, run a single forward pass and exit
    if args.smoke_test:
        try:
            imgs, masks = next(iter(train_loader))
        except Exception as e:
            print('Smoke test failed to get a batch from DataLoader:', e)
            return

        imgs = imgs.to(device)
        # masks may already be long tensor
        try:
            masks = masks.squeeze(1).long().to(device)
        except Exception:
            masks = masks.long().to(device)

        with torch.no_grad():
            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(features)

        print('Smoke test successful')
        print('  features shape:', getattr(features, 'shape', str(type(features))))
        print('  logits shape:  ', getattr(logits, 'shape', str(type(logits))))
        return

    best_val_iou = 0.0

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for epoch in range(n_epochs):
    
        # -------------------- TRAIN --------------------
        if epoch==20:
            print("Unfreezing backbone for fine-tuning...") 
            for param in backbone.parameters():
                param.requires_grad = True
        classifier.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")

        for imgs, masks in train_bar:

            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)

            with torch.no_grad():
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(features)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # -------------------- VALIDATION --------------------
        classifier.eval()
        val_loss = 0
        total_iou = 0
        total_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:

                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device)

                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(features)

                loss = criterion(logits, masks)
                val_loss += loss.item()

                iou = compute_iou(logits, masks, num_classes=n_classes)
                total_iou += iou
                total_batches += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        avg_val_iou    = total_iou / total_batches

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Val mIoU:   {avg_val_iou:.4f}")

        # Save best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(classifier.state_dict(), "best_segmentation_head.pth")
            print("Saved best model!")

    print("\nTraining complete.")
    print("Best Val mIoU:", best_val_iou)


if __name__ == '__main__':
    main()

