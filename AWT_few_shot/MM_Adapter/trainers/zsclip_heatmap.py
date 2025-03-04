import torch
import torch.nn as nn
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from load import *
from explainer import gradCAM, interpret

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

DESCRIPTION_PATH = {
    IMAGENET_DIR: '/path/to/your/dataset/', # REPLACE THIS WITH YOUR OWN PATH
    CUB_DIR: '/path/to/your/dataset/', # REPLACE THIS WITH YOUR OWN PATH
    EUROSAT_DIR: '/path/to/your/dataset/', # REPLACE THIS WITH YOUR OWN PATH
    OXFORDPET_DIR: '/home/yhiro/CoOp_/data', # REPLACE THIS WITH YOUR OWN PATH
    FOOD101_DIR: '/path/to/your/dataset/', # REPLACE THIS WITH YOUR OWN PATH
}


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.dataset = cfg.DATASET.NAME
        self.classnames = classnames
        self.description = self.get_description(self.dataset)
        self.cfg = cfg

    def encode_text(self, text):
        return self.clip_model.encode_text(text)
    
    def get_description(self, dataset):
        fpath_description = DESCRIPTION_PATH[dataset]
        # read json file
        with open(fpath_description, 'r') as f:
            description = json.load(f)
        return description

    def get_concept_features(self, classname):
        concepts = {}
        for k, v in self.description.items():
            concept = v[:5] # get top 5 concepts
            concatenated_concept = ', '.join(concept)
            rich_label = f"{wordify(k)} It may contains {concatenated_concept}"

            concept.insert(0, k) # insert class name to the first element (e.g., ['zebra', 'striped', 'mammal', 'animal', 'wildlife', 'safari'])
            tokenized_concept = self.clip_model.tokenize(concept)
            concepts[k] = tokenized_concept

        return concepts, rich_label

    def forward(self, image):
        # 画像ごとにヒートマップを作成
        # ヒートマップは各画像ごとに候補となる全記述を対象に作成
        # 作成したヒートマップは
        



@TRAINER_REGISTRY.register()
class ZeroshotCLIP_HeatMap(TrainerX):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        self.model = CustomCLIP(cfg, classnames, clip_model)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

    @torch.no_grad()
    def model_inference(self, image):
        logits = self.model(image)
        return logits

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def forward_backward(self, batch):
        images, label = self.parse_batch_train(batch)

        texts = np.array(label_to_classname)[label].tolist()

        tokenized_concepts_list = []
        rich_labels = []
        for i in range(len(texts)):
            concepts = gpt_descriptions[texts[i]][:5]
            concatenated_concepts = ', '.join(concepts)
            label = hparams['label_before_text'] + wordify(texts[i]) + hparams['label_after_text'] + " It may contains " + concatenated_concepts
            rich_labels.append(label)
            
            concepts.insert(0, texts[i])
            tokenized_concepts = clip.tokenize(concepts)
            tokenized_concepts_list.append(tokenized_concepts)

        images = images.to(self.device)
        if self.cfg.augment_text:
            rich_labels = clip.tokenize(rich_labels)
            texts = rich_labels.to(self.device)
        else:
            texts = clip.tokenize(texts)
            texts = texts.to(self.device)

        attn_map = []
        if hparams['model_size'] in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
            for k in range(len(images)):
                num_texts = tokenized_concepts_list[k].shape[0]
                repeated_image = images[k].unsqueeze(0).repeat(num_texts, 1, 1, 1)
                heatmap = gradCAM(
                    self.model.visual,
                    repeated_image,
                    self.model.encode_text(tokenized_concepts_list[k].to(self.device)),
                    getattr(self.model.visual, "layer4")
                )
                attn_map.append(heatmap)

        elif hparams['model_size'] in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']:
            for k in range(len(images)):
                R_image = interpret(model=self.model, image=images[k].unsqueeze(0), texts=tokenized_concepts_list[k].to(device), device=device)
                image_relevance = R_image[0]
                dim = int(image_relevance.numel() ** 0.5)
                R_image = R_image.reshape(-1, dim, dim)
                attn_map.append(R_image)