import random
import string

from tqdm import tqdm
import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.model import convert_weights

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


@TRAINER_REGISTRY.register()
class ZeroshotCLIP_Noise(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        noise_text = self.generate_random_text()
        noisy_temp = temp.replace("{}", f"{noise_text} {{}}")
        noisy_prompts = [noisy_temp.format(c.replace("_", " ")) for c in self.dm.dataset.classnames]
        print(noisy_prompts)
        noisy_prompts = torch.cat([clip.tokenize(p) for p in noisy_prompts]).to(self.device)
        with torch.no_grad():
            text_features_noise = clip_model.encode_text(noisy_prompts)
            text_features_noise = text_features_noise / text_features_noise.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.text_features_noise = text_features_noise
        self.clip_model = clip_model

    def generate_random_text(self, noise_length=14):
        # generate random text except for "{"  and "}"
        random_text_list = []
        while len(random_text_list) < noise_length:
            text = random.choice(string.ascii_letters + string.digits + string.punctuation)
            random_text_list.append(text)
            if "{" == text or "}" == text:
                random_text_list.remove(text)
        random_text = ''.join(random_text_list)
        return random_text
    
    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def model_inference(self, image, noise=False):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        if not noise:
            text_features = self.text_features
        else:
            text_features = self.text_features_noise
        logits = logit_scale * image_features @ text_features.t()
        return logits

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator = build_evaluator(self.cfg, lab2cname=self.lab2cname)
        self.evaluator_noise = build_evaluator(self.cfg, lab2cname=self.lab2cname)

        self.evaluator.reset()
        self.evaluator_noise.reset()

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
            output1 = self.model_inference(input, noise=False)
            output2 = self.model_inference(input, noise=True)
            self.evaluator.process(output1, label)
            self.evaluator_noise.process(output2, label)

        results1 = self.evaluator.evaluate()
        results2 = self.evaluator_noise.evaluate()

        for k, v in results1.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        for k, v in results2.items():
            tag = f"{split}/{k}_noise"
            self.write_scalar(tag, v, self.epoch)

        return list(results1.values())[0], list(results2.values())[0]