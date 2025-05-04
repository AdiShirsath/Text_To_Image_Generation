import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import re
import string
from string import digits
from typing import List

# import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import logging

# Create a logger
LOGGER = logging.getLogger("my_logger")


class EnglishProcessor:
    def __init__(self, arguments):

        self.stop_words = stopwords.words("english")
        self.digit_trans = str.maketrans("", "", digits)
        self.arguments = arguments

        # if self.arguments.use_lemmatizer:
        #     self.spacy_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        if self.arguments.use_stemmer:
            self.snowball = SnowballStemmer(language="english")

        if self.arguments.use_stemmer and self.arguments.use_lemmatizer:
            LOGGER.warning(
                "Both stemmer and lemmatizer is passed in arguments, So using lemmatizer as default"
            )
            self.arguments.use_stemmer = False

    def _remove_punctuations(self, text: str, custom_puncts: str = None):
        """
        Removes given punctuations or by default removes string.punctuation
        default - !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        Args:
            text: string to remove puncts from
            custom_puncts: list of punctuations to remove
        """
        str_trans = str.maketrans("", "", string.punctuation)
        if custom_puncts:
            puncts_to_remove = str(custom_puncts)
            str_trans = str.maketrans("", "", puncts_to_remove)

        return text.translate(str_trans)

    def _remove_digits(self, text):
        return text.translate(self.digit_trans)

    def _remove_urls(self, text):
        text = re.sub(
            r"(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?",
            "",
            text,
        )
        return text

    def _remove_emails(self, text):
        return re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", text)

    def _remove_html_tags(self, text):
        return re.sub("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", "", text)

    def _remove_stopwords(self, text):
        sent = [word for word in text.split() if word not in self.stop_words]
        return " ".join(sent)

    def _sent_lemmatizer(self, text):
        """
        Lemmatizer is text normalization process. It uses spacy to get lemmatized words
        """
        doc = self.spacy_model(text)
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas)

    def _stemmer(self, text):
        """
        Using nltk's snowball stemmer we get stemmed word for given text.

        """
        sent = text.split()
        sent = [self.snowball.stem(word) for word in sent]
        return " ".join(sent)

    def _remove_extraspaces(self, text):
        text = text.split()
        text = " ".join(text)
        return text

    # def _switcher(self, text, argument):
    #
    #     swich_dict = {
    #         'url': self.remove_urls(text),
    #         'digit': self.remove_digits(text),
    #         'punct': self.remove_punctuations(text),
    #         'stopwords': self.remove_stopwords(text),
    #         'html': self.remove_html_tags(text),
    #         'emails': self.remove_emails(text)
    #     }
    #
    #     sent = swich_dict.get(argument, lambda: 'Invalid argument')
    #     return sent

    def main(self, text: str):
        """
        Depending upon arguments provided to class, this method will preprocess text.
        Args:
            text (str): sentence
        Returns: Cleaned sentence
        """

        if len(text.split()) == 0:
            # LOGGER.error("text is not provided")
            return text

        text = text.lower()

        if self.arguments.remove_urls:
            text = self._remove_urls(text)

        if self.arguments.remove_digits:
            text = self._remove_digits(text)

        if self.arguments.remove_stopwords:
            text = self._remove_stopwords(text)

        if self.arguments.remove_punctuations:
            text = self._remove_punctuations(text)

        if self.arguments.remove_emails:
            text = self._remove_emails(text)

        if self.arguments.remove_html:
            text = self._remove_html_tags(text)

        if self.arguments.use_lemmatizer:
            text = self._sent_lemmatizer(text)

        if self.arguments.use_stemmer:
            text = self._stemmer(text)

        text = self._remove_extraspaces(text)

        return text

        
class TextImageDataset(Dataset):
    def __init__(self, image_to_captions, image_paths, transform=None):
        """
        Args:
            image_to_captions: Dict {image_path: [caption1, caption2, ...]}
            image_paths: List of image paths (absolute or relative to working dir)
            transform: Optional image transforms
        """
        self.image_to_captions = image_to_captions
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Filter out invalid paths
        self.valid_image_paths = [
            img_path for img_path in image_paths
            if os.path.exists(img_path) and img_path in image_to_captions
        ]

        # Expand dataset to (image_path, caption) pairs
        self.samples = []
        for img_path in self.valid_image_paths:
            for caption in image_to_captions[img_path]:
                self.samples.append((img_path, caption))

        print(f"Dataset size: {len(self.samples)} (all captions included)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'caption_text': caption
        }
