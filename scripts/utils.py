import clip
import torchmetrics
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
import pandas as pd
import os
import json
from pycocotools.coco import COCO


class Evaluator:

    def __init__(self, device):
        self.device = device

        # float64 doesnt work on mps so not using to(device) for fid and inception
        self.fid = FrechetInceptionDistance(feature= 2048).cpu()
        self.inception = InceptionScore().cpu()
        self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=device)
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)



    def compute_metrics(self, real_images, fake_images, captions):

        # @ convert images to unit8
        real_images = (real_images * 255).clamp(0, 255).to(torch.uint8).cpu()
        fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8).cpu()


        # Fid score

        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        fid_score = self.fid.compute()

        # Cal Inception score
        self.inception.update(fake_images)
        is_score= self.inception.compute()


        # clip score

        clip_scores = []

        for img , caption in zip(fake_images, captions):
            img = img.float() / 255.0
            img_pil = transforms.ToPILImage()(img.cpu())            
            img_clip =  self.clip_processor(img_pil).unsqueeze(0).to(self.device)

            text =  clip.tokenize([caption]).to(self.device)

            with torch.no_grad():
                image_features= self.clip_model.encode_image(img_clip)
                text_features = self.clip_model.encode_text(text)
                similarity = (image_features @ text_features.T).squeeze()
                clip_scores.append(similarity)


        clip_score = torch.mean(torch.stack(clip_scores))

        all_scores = {
            "fid": fid_score.item(),
            "inception_score": is_score[0].item(), # 0 index has mean
            "clip_score": clip_score.item()
        }
        return all_scores


    def reset(self):
        self.fid.reset()
        self.inception.reset()     
            

def combine_dataset(max_coco_samples=10000):
    import random

    # get flickr data
    flickr_path = "datasets/Flickr8k_Dataset"
    flickr_annot_path = "datasets/Flickr8k_text/Flickr8k.token.txt"
    flickr_captions = []
    with open(flickr_annot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, caption = line.split('\t')
                img_file = img_id.split('#')[0]
                full_path = os.path.join(flickr_path, img_file)
                flickr_captions.append({"image_path": full_path, "caption": caption, "source": "Flickr"})
    
    # get coco data (limit to max_coco_samples)
    coco_img_dir = "datasets/coco_dataset/train2017"
    coco_ann_file = "datasets/coco_dataset/annotations/captions_train2017.json"
    
    coco = COCO(coco_ann_file)
    coco_captions = []
    selected_img_ids = list(coco.imgs.keys())
    random.shuffle(selected_img_ids)
    selected_img_ids = selected_img_ids[:max_coco_samples]

    for img_id in selected_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        full_path = os.path.join(coco_img_dir, file_name)
        anns = coco.imgToAnns[img_id]
        for ann in anns:
            coco_captions.append({"image_path": full_path, "caption": ann['caption'], "source": "MSCOCO"})

    # combine all
    combined = pd.DataFrame(flickr_captions + coco_captions)
    combined.to_csv("datasets/combined_dataset.csv", index=False)
    print(f"✅ Combined dataset saved with {len(combined)} entries (COCO capped at {max_coco_samples})")












            
            
