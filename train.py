import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open_clip
from model_loader.clip_loader import load_clip
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# checklpoint
biomedclip_model, preprocess = open_clip.create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

class BiomedCLIPWrapper(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
    def forward(self, images, text_tokens):
        image_emb, text_emb, logit_scale = self.model(images, text_tokens)
        return image_emb, text_emb, logit_scale

clip_model = BiomedCLIPWrapper(biomedclip_model).to(device)



# frozen visual
for param in clip_model.model.visual.parameters():
    param.requires_grad = False

# check trainable
for name, param in clip_model.model.named_parameters():
    if param.requires_grad:
        print("[Trainable] ", name)
    else:
        print("[Frozen]    ", name)

# load dataset
class YourDataset(Dataset):
    def __init__(self, data_list, preprocess, tokenizer, max_length=77):
        self.data_list = data_list
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, caption = self.data_list[idx]

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)  # [3 x 224 x 224] 3 channel 

        text_tokens = self.tokenizer(caption, 
                                     truncation=True, 
                                     max_length=self.max_length, 
                                     return_tensors="pt")

        text_input_ids = text_tokens["input_ids"].squeeze(0)  # [seq_len]
        return image_tensor, text_input_ids, idx

# data loader or data list
data_list = [
    # ("/path/to/img1.png", "This is a chest X-ray showing ..."),
    # ("/path/to/img2.png", "Findings: presence of pneumonia ..."),
    # ……
]
train_dataset = YourDataset(data_list, preprocess, tokenizer, max_length=77)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# only optimize text encoder para
params_to_optimize = []
for name, param in clip_model.model.named_parameters():
    if param.requires_grad:
        params_to_optimize.append(param)

optimizer = optim.AdamW(params_to_optimize, lr=2e-5, weight_decay=0.01)

total_steps = len(train_loader) * 5  # 5 epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """
    input：
        image_embeds: [N x D]； text_embeds: [N x D]
        logit_scale:  scaler 
    return：
        average_loss = (loss_i2t + loss_t2i) / 2
    """
    # normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

    # logits
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()  # [N x N]
    logits_per_text  = logits_per_image.t()                           # [N x N]

    N = image_embeds.size(0)
    labels = torch.arange(N, device=image_embeds.device)

    loss_i2t = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2

# start training
num_epochs = 5
clip_model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (images, text_ids, indices) in enumerate(train_loader):
        images = images.to(device)                    
        text_ids = text_ids.to(device)               
        # Do we have attention_mask in the dataset? i'am not sure...
        attention_mask = text_ids.ne(0).long().to(device)

        optimizer.zero_grad()
        # forward
        image_embeds, text_embeds, logit_scale = clip_model(images, text_ids)
        # loss
        loss = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
        # backward
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / 10
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Step [{batch_idx+1}/{len(train_loader)}], "
                f"Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0

print("done")
