import os.path as osp
from copy import deepcopy
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from datasets.cls_to_names import CUSTOM_TEMPLATES, Dataset_Name_Map, get_classnames
from .ot_tools import Wasserstein_Distance


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, d_model=None, scale=1.0, down_rate=8):
        super().__init__()

        self.scale = scale
        if scale == -1.0:
            # learnable scale
            self.scale = nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=True)
        
        self.down_proj = nn.Linear(d_model, d_model // down_rate)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(d_model // down_rate, d_model)

        self.down_proj.half()
        self.up_proj.half()

        self._init_param()

    def _init_param(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        up = self.up_proj(down)

        return up * self.scale + residual


class Adapter_Learner(nn.Module):
    def __init__(self, dim=768, layer_id=[11], attn=True, mlp=True, scale=1.0, down_rate=8):
        super().__init__()

        _adapter = Adapter(dim, scale, down_rate)

        # default: both vision/langauge transformers have 12 layers
        # should be modified if more layers are used, e.g., ViT-L
        if attn:
            self.adapt_attn = nn.ModuleList([deepcopy(_adapter) if i in layer_id else nn.Identity() for i in range(12)])
        else:
            self.adapt_attn = nn.ModuleList([nn.Identity() for _ in range(12)])

        if mlp:
            self.adapt_mlp = nn.ModuleList([deepcopy(_adapter) if i in layer_id else nn.Identity() for i in range(12)])
        else:
            self.adapt_mlp = nn.ModuleList([nn.Identity() for _ in range(12)])

    def forward(self, x, layer_id = None, pos = None):
        assert pos in ['attn', 'mlp']
        if pos == 'attn':
            return self.adapt_attn[layer_id](x)
        else:
            return self.adapt_mlp[layer_id](x)


class Perturbation(nn.Module):
    def __init__(self, dim=512, layer_id=[11], aug_time=8, dtype=torch.float16):
        super().__init__()
        self.dim = dim
        self.aug_time = aug_time

        self.emb1 = nn.Linear(dim, dim, dtype=dtype)
        self.emb2 = nn.Linear(dim, dim, dtype=dtype)

        self.layer_id = layer_id

    def calc_mu_log_std(self, x):
        mu = self.emb1(x)
        log_std = self.emb2(x)
        return mu, log_std

    def forward(self, x, layer_id=None):

        if not layer_id in self.layer_id:
            return x

        self.mu, self.log_std = self.calc_mu_log_std(x)
        self.std = torch.exp(0.5 * self.log_std)

        self.std = torch.clamp(self.std, 0, 1)
        self.mu = torch.clamp(self.mu, -2, 2)

        features_aug = []
        for i in range(self.aug_time):
            # sample eps from gauss distribution N(0, 1)
            eps = torch.randn_like(self.mu)
            e = self.mu + eps * self.std
            #e = e / e.norm(dim=-1, keepdim=True)
            h = x + F.softplus(e)
            #h = h / h.norm(dim=-1, keepdim=True)
            features_aug.extend(h)
        return torch.stack(features_aug)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        # Adapter module for vision transformer
        if cfg.Adapter.Visual:
            self.visual_adapter_learner = Adapter_Learner(
                    clip_model.visual.ln_post.weight.shape[0],
                    cfg.Adapter.Layer_ID, cfg.Adapter.Attn, cfg.Adapter.MLP, 
                    cfg.Adapter.Scale, cfg.Adapter.Down_Rate
                )
        else:
            self.visual_adapter_learner = None

        # Adapter module for text transformer
        if cfg.Adapter.Text:
            self.text_adapter_learner = Adapter_Learner(
                    clip_model.ln_final.weight.shape[0],
                    cfg.Adapter.Layer_ID, cfg.Adapter.Attn, cfg.Adapter.MLP, 
                    cfg.Adapter.Scale, cfg.Adapter.Down_Rate
                )
        else:
            self.text_adapter_learner = None

        ndim_visual = clip_model.visual.ln_post.weight.shape[0]
        ndim_text = clip_model.ln_final.weight.shape[0]

        layer_id = [11]

        self.perturb_visual = Perturbation(ndim_visual, layer_id, 8, dtype=clip_model.dtype)
        self.perturb_text = Perturbation(ndim_text, layer_id, 8, dtype=clip_model.dtype)

        self.adapter_learners = nn.ModuleDict({
                "visual_adapter_learner": self.visual_adapter_learner,
                "text_adapter_learner": self.text_adapter_learner,
                "perturb_visual": self.perturb_visual,
                "perturb_text": self.perturb_text,
            })

        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.desc_per_batch = cfg.LLM.Desc_Per_Batch
        self.tot_desc = cfg.LLM.Num_desc + 1
        self.classnames = classnames
        self.texts = self.get_text_input(cfg) # n_cls x n_desc+1 x 77


    @torch.no_grad()
    def get_text_input(self, cfg):
        dataset_name = Dataset_Name_Map[cfg.DATASET.NAME]
        if dataset_name == 'imagenetv2':
            classnames = self.classnames
        else:
            classnames = get_classnames(dataset_name)
        
        self.n_cls = len(classnames)
        # get LLM descriptors
        if dataset_name in ['imagenetv2', 'imagenet_a']:
            description_file = osp.join(cfg.LLM.PATH, f'imagenet.json')
        else:
            description_file = osp.join(cfg.LLM.PATH, f'{dataset_name}.json')
        print(f'Using description file: {description_file}')
        llm_descriptions = json.load(open(description_file))

        template = CUSTOM_TEMPLATES[dataset_name]
        prompts = []
        for classname in classnames:
            prompt = template.format(classname.replace("_", " "))
            prompts.append(prompt + '.')
            assert len(llm_descriptions[classname]) >= cfg.LLM.Num_desc
            for i in range(cfg.LLM.Num_desc):
                prompt_desc = prompt + '. ' + llm_descriptions[classname][i]
                prompts.append(prompt_desc)
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        return prompts.reshape(len(classnames), cfg.LLM.Num_desc+1, 77).cuda()


    def forward(self, image):
        # forward function for training
        # image: batch size x aug time x 3 x 224 x 224
        bs, aug_time, _, _, _ = image.shape
        aug_time = 8
        self.desc_per_batch = 1
        image = image.reshape(-1, 3, 224, 224)
        # teacher model
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # student model
        image_features_aug = self.clip_model.encode_image(image, self.visual_adapter_learner, self.perturb_visual)
        image_features_aug = image_features_aug / image_features_aug.norm(dim=-1, keepdim=True)
        
        # sample sub-set texts for efficient training
        sample_idx = torch.randperm(self.texts.size(1)-1)[:self.desc_per_batch-1]
        sub_texts = torch.cat([
                self.texts[:, :1], self.texts[:, 1:][:, sample_idx]
            ], dim=1).reshape(-1, 77)
        
        # teacher model
        with torch.no_grad():
            text_features = self.clip_model.encode_text(sub_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # student model
        text_features_aug = self.clip_model.encode_text(sub_texts, self.text_adapter_learner, self.perturb_text)
        text_features_aug = text_features_aug / text_features_aug.norm(dim=-1, keepdim=True)


        return image_features_aug, text_features_aug, image_features, text_features


    @torch.no_grad()
    def get_text_features(self, ):
        text_features = self.clip_model.encode_text(self.texts.reshape(-1, 77), self.text_adapter_learner)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


    @torch.no_grad()
    def inference(self, image, text_features):
        # forward function for testing
        # image: batch size x aug time x 3 x 224 x 224
        bs, aug_time, _, _, _ = image.shape
        image = image.reshape(-1, 3, 224, 224)
        image_features = self.clip_model.encode_image(image, self.visual_adapter_learner)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # (bs x aug_time) x (n_cls x self.tot_desc)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        logits = logits.reshape(bs, aug_time, self.n_cls, self.tot_desc).permute(0, 1, 3, 2).contiguous()
        wass_dist = Wasserstein_Distance(logits, self.logit_scale, False)

        return wass_dist


@TRAINER_REGISTRY.register()
class AWT_PERTUBATION(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter_learner" not in name:
                param.requires_grad_(False)

        # Double check
        # comment out because there are too many parameters
        # enabled = set()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)
        # print(f"Parameters to be updated: {enabled}")

        print("Trainable params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter_learners, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give adapter_learner to the optimizer
        self.optim = build_optimizer(self.model.adapter_learners, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learners", self.model.adapter_learners, self.optim, self.sched)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        image_features_aug, text_features_aug, image_features, text_features = self.model(image)

        # image: batch size x aug time x 3 x 224 x 224
        bs, aug_time, _, _, _ = image.shape
        aug_time = 8
        # (bs x aug_time) x (n_cls x self.desc_per_batch)
        logits = self.model.logit_scale.exp() * image_features_aug @ text_features_aug.t()
        logits = logits.reshape(bs, aug_time, self.model.n_cls, aug_time).permute(0, 1, 3, 2).contiguous()
        wass_dist = Wasserstein_Distance(logits, self.model.logit_scale, True) # bs x n_cls

        loss_ce = F.cross_entropy(wass_dist, label)
        
        loss_pertubation_img = F.l1_loss(image_features, image_features_aug.view(bs, aug_time, -1).mean(dim=1), 
                                         reduction='mean')
        loss_pertubation_text = F.l1_loss(text_features, text_features_aug.view(bs, aug_time, -1).mean(dim=1),
                                          reduction='mean')

        loss = loss_ce - loss_pertubation_img - loss_pertubation_text

        self.model_backward_and_update(loss, names="adapter_learners")

        loss_summary = {
            "loss_ce": loss_ce.item(),
            "loss_pertubation_img": loss_pertubation_img.item(),
            "loss_pertubation_text": loss_pertubation_text.item(),
            "acc": compute_accuracy(wass_dist, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = torch.stack(input, dim=0) # aug_time x batch_size x 3 x 224 x 224
        input = torch.transpose(input,0,1).contiguous() # batch_size x aug_time x 3 x 224 x 224
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = torch.stack(input, dim=0) # aug_time x batch_size x 3 x 224 x 224
        input = torch.transpose(input,0,1).contiguous() # batch_size x aug_time x 3 x 224 x 224
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def test(self, split=None):
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

        # prepare text features
        text_features = self.model.get_text_features()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model.inference(input, text_features)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        # In AWT, we just pick the last-epoch ckpt for evaluation
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)