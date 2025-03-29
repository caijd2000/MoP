import torch
import timm
import json
import vit
import open_clip
from transformers import CLIPProcessor, CLIPModel

def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def get_model(encoder_type):
    encoder = 0
    if encoder_type == 'uni':
        encoder = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        encoder.load_state_dict(torch.load("./UNI/pytorch_model.bin", map_location="cpu"), strict=True)

    elif encoder_type == 'gigapath':
        cfg = load_cfg_from_json("/public_bme/data/v-caijd/checkpoints/checkpoints/prov-gigapath/config.json")
        encoder = timm.create_model("vit_giant_patch14_dinov2",**cfg['model_args'],pretrained_cfg=cfg['pretrained_cfg'], dynamic_img_size=True, pretrained=False,checkpoint_path="./prov-gigapath/pytorch_model.bin")

    elif encoder_type == 'conch':
        from open_clip_custom import create_model_from_pretrained
        model,process = create_model_from_pretrained('conch_ViT-B-16',checkpoint_path='./CONCH/pytorch_model.bin')
        encoder = model.visual

    elif encoder_type == "plip":
        clip_model = CLIPModel.from_pretrained(pretrained_model_name_or_path = './plip_model/')
        processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path = './plip_model/')

        return clip_model, processor

    else:
        print("Type no found!")

    return encoder