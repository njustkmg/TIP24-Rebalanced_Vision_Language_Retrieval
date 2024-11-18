import shutil
import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.model_retrieval import XVLM
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from metric import get_score

auto_init_temp = 0.07


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['load_pretrain'] is True:
        metric_logger.add_meter('loss_iic', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_ttc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_kdi', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_kdt', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100
    step_size = 100

    for i, (image1, text1, idx, single_text_embed, single_image_embed) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image1 = image1.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        single_text_embed = single_text_embed.to(device, non_blocking=True)
        single_image_embed = single_image_embed.to(device, non_blocking=True)
        text_input1 = tokenizer(text1, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)

        if config['load_pretrain'] is True:
            loss_itc, loss_itm, loss_iic, loss_ttc, loss_kd, loss_kdi, loss_kdt = model(image1, text_input1.input_ids, text_input1.attention_mask, idx=idx,
                                                single_text_embed=single_text_embed, single_image_embed=single_image_embed)
            loss = config['a'] * loss_itc + config['b'] * loss_itm + config['c'] * loss_iic + config['d'] * loss_ttc + config['e'] * loss_kd + config['f'] * loss_kdi + config['g'] * loss_kdt
        else:
            loss_itc, loss_itm = model(image1, text_input1.input_ids, text_input1.attention_mask, idx=idx)
            loss = config['a'] * loss_itc + config['b'] * loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        if config['load_pretrain'] is True:
            metric_logger.update(loss_iic=loss_iic.item())
            metric_logger.update(loss_ttc=loss_ttc.item())
            metric_logger.update(loss_kd=loss_kd.item())
            if config['distance'] != "wd":
                metric_logger.update(loss_kdi=loss_kdi.item())
                metric_logger.update(loss_kdt=loss_kdt.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_embeds = []
    text_feats = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []       # 1024
    image_embeds = []      # 256
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.vision_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    # i2t
    print("-----------------i2t")
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    # t2i
    print("-----------------t2i")
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[i].repeat(config['k_test'], 1, 1),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
        
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), image_feats.detach().cpu().numpy(), image_embeds.detach().cpu().numpy(), text_feats.detach().cpu().numpy(), text_embeds.detach().cpu().numpy()


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Creating model", flush=True)
    model = XVLM(config=config)
    if args.evaluate:
        model.load_pretrained(config['checkpoint'], config, is_eval=args.evaluate)
    else:
        if args.load_pretrain:
            path = "output/" + args.dataname + "/pretrain/checkpoint_best.pth"
            model.load_pretrained(path, config, is_eval=args.evaluate)
    if config['temp'] != auto_init_temp:
        model.temp = config['temp']
        model.temp2 = config['temp2']
    model = model.to(device)

    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset, test_dataset = create_dataset('re', config, args.evaluate)

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]

        score_test_i2t, score_test_t2i, image_feats, image_embeds, text_feats, text_embeds = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            test_result = get_score(score_test_i2t, score_test_t2i, image_feats, image_embeds, text_feats, text_embeds, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt, config["dataname"], 256)
            print(test_result)

        dist.barrier()

    else:
        print("Start training", flush=True)

        train_dataset_size = len(train_dataset)

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                              batch_size=[config['batch_size_train']] + [
                                                                  config['batch_size_test']] * 2,
                                                              num_workers=[8, 4, 4],
                                                              is_trains=[True, False, False],
                                                              collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            score_test_i2t, score_test_t2i, image_feats, image_embeds, text_feats, text_embeds = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                test_result = get_score(score_test_i2t, score_test_t2i, image_feats, image_embeds, text_feats, text_embeds, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt, config["dataname"], 1024)

                print(test_result)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch}

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if test_result['r_mean'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = test_result['r_mean']
                    best_epoch = epoch
                
                elif epoch >= config['schedular']['epochs'] - 1:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # (1) when pretrain student models using ITC and ITM, set "load_pretrain = False"; (2) load the pretrained model and train with all objectives (our method), set "load_pretrain = True"
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--dataname', default='flickr')  # flickr lamda = 0.8547  vizwiz lamda = 
    parser.add_argument('--output_dir', type=str, default="sample")  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:5275', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--local_rank', default=1)
    parser.add_argument('--coco_1k', default=0)
    parser.add_argument('--save_embed', type=int, default=0)
    parser.add_argument('--embed', type=str, default='tmp')

    parser.add_argument('--bs', default=36, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--auto', default=1, type=int)  # False 0   true 1
    parser.add_argument('--lamda', default=0, type=float)
    parser.add_argument('--distance', default='mae', type=str)
    parser.add_argument('--a', default=1.0, type=float)
    parser.add_argument('--b', default=2.0, type=float)
    parser.add_argument('--c', default=1.0, type=float)
    parser.add_argument('--d', default=2.0, type=float)
    parser.add_argument('--e', default=2.0, type=float)
    parser.add_argument('--f', default=1.0, type=float)
    parser.add_argument('--g', default=1.0, type=float)
    parser.add_argument('--temp', default=0.07, type=float)
    parser.add_argument('--temp2', default=0.07, type=float)
    args = parser.parse_args()

    args.seed = torch.randint(low=0, high=10000000, size=(1,))
    args.output_dir = "output/" + args.dataname + "/" + args.output_dir
    
    config = yaml.load(open("configs/Retrieval_" + args.dataname + ".yaml", 'r'), Loader=yaml.Loader)
    config['load_pretrain'] = args.load_pretrain
    config['checkpoint'] = args.output_dir + "/checkpoint_best.pth"
    config['dataname'] = args.dataname
    config['save_embed'] = args.save_embed
    config['coco_1k'] = int(args.coco_1k)
    config['embed'] = args.output_dir.split("/")[-1]
    config["batch_size_train"] = args.bs
    config['auto'] = args.auto
    config['lamda'] = args.lamda
    config['distance'] = args.distance
    config['a'] = args.a
    config['b'] = args.b
    config['c'] = args.c
    config['d'] = args.d
    config['e'] = args.e
    config['f'] = args.f
    config['g'] = args.g
    config['temp']= args.temp
    config['temp2']= args.temp2
    config['n_layers'] = args.n_layers
    config['n_heads'] = args.n_heads

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    optim = utils.AttrDict(config["optimizer"])
    sche = utils.AttrDict(config["schedular"])
    main(args, config)

'''
nohup sh train_coco.sh > logs/coco.txt 2>&1 &
nohup sh train_flickr.sh > logs/flickr.txt 2>&1 &
nohup sh train_vizwiz.sh > logs/vizwiz.txt 2>&1 &

'''