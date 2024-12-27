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

import os
import pickle

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
            nn.init.zeros_(self.up_proj.weight)
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


class GaussianProjector(nn.Module):
    def __init__(self, n_dim, latent_dim=512, num_classes=10):
        super(GaussianProjector, self).__init__()
        self.n_dim = n_dim
        self.num_classes = num_classes
        self.vae_encoder = nn.Sequential(
            nn.Linear(n_dim, n_dim * 2 * self.num_classes),
            nn.GELU(),
            nn.Linear(n_dim * 2 * self.num_classes, n_dim * 2 * self.num_classes)
        ).half()

        self._init_param()
    
    def _init_param(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    # def encode(self, x, c):
    #     stats = self.vae_encoder(x)
    #     mean, logvar = stats.chunk(2, dim=-1)
    #     return mean, logvar

    def encode(self, x, c):
        batch_size = x.size(0)
        stats = self.vae_encoder(x)
        stats = stats.view(batch_size, self.num_classes, -1) # bs x n_cls x 2 * n_dim
        mean, logvar = stats.chunk(2, dim=-1) # bs x n_cls x n_dim

        if c is None:
            return mean, logvar

        # select class-specific mean and logvar
        c = c.view(-1, 1, 1).expand(-1, 1, mean.size(-1)) 
        mean = mean.gather(1, c).squeeze(1)
        logvar = logvar.gather(1, c).squeeze(1)
        return mean, logvar

    def forward(self, x, c=None):
        mean, logvar = self.encode(x, c)

        if not self.training:
            if c is None:
                return mean

        if self.training:
            z = self.reparameterize(mean, logvar)
            kl_loss = self.kl_divergence(mean, logvar)
            return z, mean, logvar, kl_loss


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

        ndim = clip_model.ln_final.weight.shape[0]
        self.noise_modulator_visual = GaussianProjector(ndim)
        self.noise_modulator_text = GaussianProjector(ndim)

        self.adapter_learners = nn.ModuleDict({
                "visual_adapter_learner": self.visual_adapter_learner,
                "text_adapter_learner": self.text_adapter_learner,
            })
        
        self.noise_modulator = nn.ModuleDict({
            "noise_modulator_visual": self.noise_modulator_visual,
            "noise_modulator_text": self.noise_modulator_text,
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
        image = image.reshape(-1, 3, 224, 224)

        # student model
        image_features = self.clip_model.encode_image(image, self.visual_adapter_learner) # (bs x aug_time) x 512
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # sample sub-set texts for efficient training
        sample_idx = torch.randperm(self.texts.size(1)-1)[:self.desc_per_batch-1]
        sub_texts = torch.cat([
                self.texts[:, :1], self.texts[:, 1:][:, sample_idx]
            ], dim=1).reshape(-1, 77) # (n_cls x desc_per_batch) x 77
        
        # student model
        text_features = self.clip_model.encode_text(sub_texts, self.text_adapter_learner) # (n_cls x desc_per_batch) x 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
    
    def vae(self, image_features, text_features, label):
        # image: (bs x aug_time) x 512
        # text: (n_cls x desc_per_batch) x 512
        # label: bs x 1

        bs = label.size(0)
        aug_time = image_features.size(0) // bs

        # label: bs x 1 -> (bs x aug_time) x 1
        label_image = label.unsqueeze(1).repeat(1, aug_time).reshape(-1).to(label.device)
        # create text label. (n_cls x desc_per_batch) x 1
        label_text = torch.arange(self.n_cls).unsqueeze(1).repeat(1, self.desc_per_batch).reshape(-1).to(label.device)

        image_features_z, _, _, kl_loss_img = self.noise_modulator_visual(image_features, label_image)
        text_features_z, _, _, kl_loss_text = self.noise_modulator_text(text_features, label_text)

        image_features_z = image_features_z.view(bs, aug_time, -1) # bs x aug_time x 512
        text_features_z = text_features_z.view(self.n_cls, self.desc_per_batch, -1) # n_cls x desc_per_batch x 512

        mean_visual = image_features_z.mean(dim=1) # bs x 512
        mean_text = text_features_z.mean(dim=1) # n_cls x 512

        # centering
        image_features_center = image_features_z - mean_visual.unsqueeze(1) # bs x aug_time x 512
        text_features_center = text_features_z - mean_text.unsqueeze(1)

        # calculate covariance matrix (divide by (B-1) for Beesel correction)
        cov_visual = (image_features_center.transpose(1, 2) @ image_features_center) / (aug_time - 1) # bs x 512 x 512
        cov_text = (text_features_center.transpose(1, 2) @ text_features_center) / (self.desc_per_batch - 1) # n_cls x 512 x 512

        wass_loss = 0
        for i, l in enumerate(label):
            m1 = mean_visual[i]
            m2 = mean_text[l]

            C1 = cov_visual[i].to(torch.float32)
            C2 = cov_text[l].to(torch.float32)

            # calculate 2-Wasserstein distance
            diff_mean_sq = torch.sum((m1 - m2).pow(2))

            e2, U2 = torch.linalg.eigh(C2)
            e2_sqrt = torch.sqrt(torch.clamp(e2, min=1e-7))
            C2_half = (U2 * e2_sqrt.unsqueeze(0)) @ U2.T

            temp = C2_half @ C1 @ C2_half
            e_temp, U_temp = torch.linalg.eigh(temp)
            e_temp_sqrt = torch.sqrt(torch.clamp(e_temp, min=1e-7))
            trace_term = torch.sum(e_temp_sqrt)

            trace_c1_c2 = torch.trace(C1) + torch.trace(C2)

            wass_loss += diff_mean_sq + trace_c1_c2 - 2.0 * trace_term
        
        wass_loss = wass_loss.mean()

        image_features_z = image_features_z.view(bs*aug_time, -1)
        text_features_z = text_features_z.view(self.n_cls*self.desc_per_batch, -1)

        return image_features_z, text_features_z, kl_loss_img, kl_loss_text, wass_loss

    @torch.no_grad()
    def get_text_features(self, ):
        text_features = self.clip_model.encode_text(self.texts.reshape(-1, 77), self.text_adapter_learner)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_text_features_gauss(self,):
        text_features = self.clip_model.encode_text(self.texts.reshape(-1, 77), self.text_adapter_learner)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        label_text = torch.arange(self.n_cls).unsqueeze(1).repeat(1, self.desc_per_batch).reshape(-1).to(text_features.device)
        self.noise_modulator_text.eval()
        text_features_z, _, _ = self.noise_modulator_text.encode(text_features, label_text)

        return text_features_z

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

    @torch.no_grad()
    def inference_(self, image, text_features_z):
        bs, aug_time, _, _, _ = image.shape
        image = image.reshape(-1, 3, 224, 224)
        image_features = self.clip_model.encode_image(image, self.visual_adapter_learner)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_features_z = self.noise_modulator_visual(image_features)

        image_features_z = image_features_z.view(-1, 512)

        logits = image_features_z @ text_features_z.T # (bs x aug_time) x (Num_desc x n_cls)

        dist = torch.cdist(image_features_z, text_features_z)
        idx = dist.argmin(dim=1).item()


@TRAINER_REGISTRY.register()
class AWT_RESAMPLE(TrainerX):

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
            if "noise_modulator" in name:
                param.requires_grad_(True)

        print("Trainable params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter_learners, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give adapter_learner to the optimizer
        self.optim = build_optimizer(self.model.adapter_learners, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learners", self.model.adapter_learners, self.optim, self.sched)

        self.optim_noise = build_optimizer(self.model.noise_modulator, cfg.OPTIM)
        self.sched_noise = build_lr_scheduler(self.optim_noise, cfg.OPTIM)
        self.register_model("noise_modulator", self.model.noise_modulator, self.optim_noise, self.sched_noise)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        bs, aug_time, _, _, _ = image.shape

        image_features, text_features = self.model(image)
        image_features_z, text_features_z, kl_loss_img, kl_loss_text, wass_loss = self.model.vae(image_features, text_features, label)

        # logits = self.model.logit_scale.exp() * image_features @ text_features.t()
        # logits = logits.reshape(bs, aug_time, self.model.n_cls, self.model.desc_per_batch).permute(0, 1, 3, 2).contiguous()
        # output = Wasserstein_Distance(logits, self.model.logit_scale, True) # bs x n_cls
        #import pdb; pdb.set_trace()
        output = torch.cdist(image_features_z, text_features_z) # (bs x aug_time) x (desc_per_batch x n_cls)
        output = output.reshape(bs, aug_time, self.model.n_cls, self.model.desc_per_batch)
        
        output = output.min(dim=1)[0]
        output = output.min(dim=-1)[0] # [1, bs x n_cls]
        output = output.reshape(bs, self.model.n_cls) # # bs x n_cls
        output = -output

        loss_ce = F.cross_entropy(output, label)
        wass_loss = wass_loss * 0.0001
        kl_loss_img = kl_loss_img * 0.1
        kl_loss_text = kl_loss_text * 0.1
        
        #loss = loss_ce
        loss = kl_loss_img + kl_loss_text + loss_ce + wass_loss
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_ce": loss_ce.item(),
            "kl_loss_img": kl_loss_img.item(),
            "kl_loss_text": kl_loss_text.item(),
            "wass_loss": wass_loss.item(),
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # save pickle file for debugging
        image_features = image_features.cpu().detach().numpy()
        text_features = text_features.cpu().detach().numpy()
        image_features_z = image_features_z.cpu().detach().numpy()
        text_features_z = text_features_z.cpu().detach().numpy()
        label_out = label.cpu().detach().numpy()

        with open(os.path.join(self.cfg.OUTPUT_DIR, f'img_feat_stu_{self.epoch}.pkl'), 'wb') as f:
            pickle.dump(image_features, f)
        with open(os.path.join(self.cfg.OUTPUT_DIR, f'text_feat_stu_{self.epoch}.pkl'), 'wb') as f:
            pickle.dump(text_features, f)
        with open(os.path.join(self.cfg.OUTPUT_DIR, f'img_feat_mod_{self.epoch}.pkl'), 'wb') as f:
            pickle.dump(image_features_z, f)
        with open(os.path.join(self.cfg.OUTPUT_DIR, f'text_feat_mod_{self.epoch}.pkl'), 'wb') as f:
            pickle.dump(text_features_z, f)
        with open(os.path.join(self.cfg.OUTPUT_DIR, f'label_{self.epoch}.pkl'), 'wb') as f:
            pickle.dump(label_out, f)

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