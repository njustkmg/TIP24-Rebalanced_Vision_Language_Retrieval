import torch
from models.xvlm import XVLMBase, load_pretrained


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads  # 12
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx=None, single_text_embed=None, single_image_embed=None):
        image_embeds, image_atts = self.get_vision_embeds(image)     # img_pre [bs, 145, 1024], [bs, 145]
        text_embeds = self.get_text_embeds(text_ids, text_atts)     # txt_pre [bs, 20, 768]

        # pretrain
        if single_image_embed is None:
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)    # img, txt
            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts,
                                                    text_feat, idx=idx, single_text=single_text_embed,
                                                    single_image=single_image_embed)
            return loss_itc, loss_itm
        else:
            image_feat, text_feat, img_sig, txt_sig = self.get_features(image_embeds, text_embeds, single_image_embed, single_text_embed)    # img, txt
            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_iic = self.get_single_contrast(image_feat, img_sig, idx=idx)
            loss_ttc = self.get_single_contrast(text_feat, txt_sig, idx=idx)
            loss_itm, loss_kd, loss_kdi, loss_kdt = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts,
                                                    text_feat, idx=idx, single_text=single_text_embed,
                                                    single_image=single_image_embed)
            return loss_itc, loss_itm, loss_iic, loss_ttc, loss_kd, loss_kdi, loss_kdt
