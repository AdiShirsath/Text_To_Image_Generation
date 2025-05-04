import torch
from scripts import eng_processor
from transformers import BertTokenizer



def generate_image_from_text(text, text_encoder, generator, device):

    # preprocessing of text
    processed_text = eng_processor.main(text)

    # tokenize with bert
    tokenizer= BertTokenizer.from_pretrained("bert-base-uncased")

    tokens=tokenizer(
        processed_text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    # generate image
    with torch.no_grad():
        text_encoder.eval()
        generator.eval()

        text_embedding =  text_encoder(tokens["input_ids"], tokens["attention_mask"])
        gen_image= generator(text_embedding)
        gen_image = 0.5 * gen_image + 0.5 # [-1, 1] --> [0,1]
        gen_image = gen_image.squeeze(0).cpu().permute(1,2,0).numpy()

    return gen_image