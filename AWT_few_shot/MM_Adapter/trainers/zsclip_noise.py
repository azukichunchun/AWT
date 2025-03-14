import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
from clip.simple_tokenizer import SimpleTokenizer

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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

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

        self.text_features = text_features
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        self.encode_text_from_embedding = TextEncoder(clip_model)
        self.tokenizer = SimpleTokenizer()

        self.adv_prompts = self.find_adversarial_prompt()
        print("Adversarial (悪意のある) プロンプト:", self.adv_prompts)

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def find_adversarial_prompt(self, num_steps=100, lr=0.1, noise_std=0.01):

        batch = next(iter(self.train_loader_x))
        images, labels = self.parse_batch_train(batch)

        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        baseline_prompts = [temp.format(c.replace("_", " ")) for c in self.dm.dataset.classnames]
        baseline_tokens = torch.cat([clip.tokenize(p) for p in baseline_prompts]).to(self.device) # shape: (num_classes, 77)
        
        adv_emb = self.clip_model.token_embedding(baseline_tokens).type(self.dtype)  # shape: (num_classes, 77, 512)
        adv_emb = adv_emb.detach().clone().requires_grad_(True)

        token_embedding = self.clip_model.token_embedding.weight.detach().clone()

        optimizer = torch.optim.SGD([adv_emb], lr=lr)
        for step in range(num_steps):
            text_features_adv = self.encode_text_from_embedding(adv_emb, baseline_tokens)
            text_features_adv = text_features_adv / text_features_adv.norm(dim=-1, keepdim=True)

            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features_adv.t()
            loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Langevin dynamics
            with torch.no_grad():
                adv_emb.add_(torch.randn_like(adv_emb) * noise_std)

            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item()}")

        # 連続空間上の最適化後の埋め込みを，最も近い離散トークン列に射影（AutoPrompt の手法）
        adv_token_ids = self.project_embeddings_to_tokens(adv_emb, token_embedding)
        adv_prompts = self.decode_tokens(adv_token_ids)
        return adv_prompts

    def project_embeddings_to_tokens(self, adv_emb, token_embedding):
        """
        adv_emb: (num_prompts, seq_length, embed_dim)
        token_embedding: (vocab_size, embed_dim)
        """
        num_prompts, seq_length, embed_dim = adv_emb.shape
        adv_token_ids = torch.zeros((num_prompts, seq_length), dtype=torch.long, device=adv_emb.device)

        adv_emb_norm = adv_emb / adv_emb.norm(dim=-1, keepdim=True)
        token_embedding_norm = token_embedding / token_embedding.norm(dim=-1, keepdim=True)
        token_embedding_norm = token_embedding_norm.type(adv_emb.dtype)
        
        for i in range(num_prompts):
            for j in range(seq_length):
                # 各トークン位置について全語彙とのコサイン類似度を計算
                sims = torch.matmul(token_embedding_norm, adv_emb_norm[i, j])
                adv_token_ids[i, j] = torch.argmax(sims)
        return adv_token_ids

    def decode_tokens(self, token_ids):
        prompts = []
        for ids in token_ids:
            tokens = [self.tokenizer.decoder.get(token.item(), "") for token in ids if token != 0]
            prompts.append(" ".join(tokens).strip().replace("</w>", ""))
        import pdb; pdb.set_trace()
        return prompts

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
