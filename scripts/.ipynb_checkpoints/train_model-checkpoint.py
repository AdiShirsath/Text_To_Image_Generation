import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer



class BertEncoder(nn.Module):

    def __init__(self, latent_dim=256):

        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze bert layer for saving memory
        for param in self.bert.parameters():
            param.requires_grad =False
        # adding drop out for regularization
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, latent_dim)


    def forward(self, input_ids, attention_mask):

        # use torch for reducing memory use
        with torch.no_grad():  
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)



# ref for gan generator using pytroch: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class Generator(nn.Module):
    def __init__(self, embedding_dim= 256, img_dim=256, noise_dim=100):
        super(Generator, self).__init__()

        self.fc = nn.Linear(embedding_dim + noise_dim, 512 * 4* 4)

        self.conv_blocks = nn.Sequential(

            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # use relu in gan for avoiding noise and used in original gan
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, text_embedding, noise):
        x = torch.cat((text_embedding, noise), dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        return self.conv_blocks(x)
        
        

# discriminator to envaluate image, checks how image aligns with text

class Discriminator(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(256, 512, 4, 2, 1),  # 32 → 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),  # 16 → 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )

        # Compute feature size dynamically to avoide size issue
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.conv_blocks(dummy_input)
            self.linear_input_features = dummy_output.view(1, -1).shape[1]

        self.text_fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(self.linear_input_features + 512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        # Ensure the image is resized to 256x256
        if img.shape[-1] != 256 or img.shape[-2] != 256:
            img = nn.functional.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
    
        img_features = self.conv_blocks(img)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten image features
    
        text_features = self.text_fc(text_embedding)
    
        combined = torch.cat((img_features, text_features), dim=1)
        validity = self.final_fc(combined)
        return validity









        