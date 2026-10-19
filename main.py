import os
import torchvision
from torch.utils.data.dataloader import default_collate
from copy import deepcopy
import utils
import random
import numpy as np
import time
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.cuda import amp
import argparse
import transreid
import torch.nn as nn
import logging
from loss import TripletLoss_ADP, KLDivLoss
from torchvision import transforms
import transform
import data
import math
import datetime
from metric import eval_regdb, eval_sysu
from torch.utils.tensorboard import SummaryWriter

global checkpoint


def fix_state(seed):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)


def init_logging(args):
    output_dir = os.path.join(args.output_dir, f'{args.dataset}')
    output_dir = os.path.join(output_dir, f'{args.model}_{args.backbone}_b{args.batch_size}_n{args.num_pos}'
                                          f'_wd{args.weight_decay}_lr{args.lr}_square{args.square}'
                                          f'_stride{args.stride}_siecoe{args.siecoe}_width{args.width}'
                                          f'_height{args.height}_camera{args.camera_num}_shuffle{args.shuffle}'
                                          f'_shift{args.shift}_divide{args.divide}')

    if not args.cos:
        output_dir += f'_steplr{args.lr_gamma}'
        for milestone in args.milestone:
            output_dir += f'_{milestone}'
    else:
        output_dir += f'_coslr{args.epochs}'

    if args.nonlinear:
        output_dir += '_nonlinear'

    if args.rearrange:
        output_dir += '_rearange'

    if args.local:
        output_dir += '_local'

    if args.warmup > 0:
        output_dir += f'_warmup{args.warmup + 1}'

    if args.ls > 0:
        output_dir += f'_ls{args.ls}'

    if args.adam:
        output_dir += f'_adam'
    elif args.adamw:
        output_dir += f'_adamw'
    else:
        output_dir += f'_sgd'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(os.path.join(output_dir, args.output_log))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # attach both handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger, output_dir


def train_one_epoch(model, criterion_id, criterion_tri, optimizer, data_loader, device,
                    epochs, args, scaler, logger, criterion_kl):
    model.train()
    metric_logger = utils.MetricLogger(logger, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epochs)

    for color1, color2, infrare, color_target, infrare_target, color_cam, infrare_cam in \
            metric_logger.log_every(data_loader, args.print_freq, header):
        start = time.time()
        correct = 0

        if color_cam[0] > 0:
            color_cam, infrare_cam = color_cam.to(device), infrare_cam.to(device)

        if args.tri:
            color1, color2, infrare, color_target, infrare_target = \
                color1.to(device), color2.to(device), infrare.to(device), \
                color_target.to(device), infrare_target.to(device)
            labels = torch.cat((color_target, color_target, infrare_target), 0)
            color = torch.cat((color1, color2), 0)
            if color_cam[0] > 0:
                cams = torch.cat((color_cam, color_cam, infrare_cam), 0)
            else:
                cams = None
        else:
            color1, infrare, color_target, infrare_target = \
                color1.to(device), infrare.to(device), \
                color_target.to(device), infrare_target.to(device)
            labels = torch.cat((color_target, infrare_target), 0)
            color = color1
            if color_cam[0] > 0:
                cams = torch.cat((color_cam, infrare_cam), 0)
            else:
                cams = None

        if scaler is not None:
            with amp.autocast():
                if not args.two:
                    feature, output = model(torch.cat((color, infrare), 0), cam_label=cams)
                else:
                    feature, output = model(color, infrare, cam_label=cams)

                if isinstance(output, list):
                    loss_id = [criterion_id(score, labels) for score in output[1:]]
                    loss_id = sum(loss_id) / len(loss_id)
                    loss_id = 0.5 * loss_id + 0.5 * criterion_id(output[0], labels)

                    if args.tri:
                        n = 3
                    else:
                        n = 2

                    loss_kl = []
                    for out in output[1:]:
                        output1 = out.narrow(0, 0, out.shape[0] // n)
                        output2 = out.narrow(0, (n - 1) * out.shape[0] // n, out.shape[0] // n)
                        loss_kl.append(criterion_kl(output1, output2))

                    output1 = output[0].narrow(0, 0, output[0].shape[0] // n)
                    output2 = output[0].narrow(0, (n - 1) * output[0].shape[0] // n, output[0].shape[0] // n)
                    loss_kl = sum(loss_kl) / len(loss_kl)
                    loss_kl = 0.5 * loss_kl + 0.5 * criterion_kl(output1, output2)

                else:
                    loss_id = criterion_id(output, labels)

                    if args.tri:
                        n = 3
                    else:
                        n = 2
                    output1 = output.narrow(0, 0, output.shape[0] // n)
                    output2 = output.narrow(0, (n - 1) * output.shape[0] // n, output.shape[0] // n)
                    loss_kl = criterion_kl(output1, output2)

                if isinstance(feature, list):
                    loss_tri = [criterion_tri(feat, labels, args.norm)[0] for feat in feature[1:]]
                    loss_tri = sum(loss_tri) / len(loss_tri)
                    loss_tri = 0.5 * loss_tri + 0.5 * criterion_tri(feature[0], labels)[0]
                else:
                    loss_tri = criterion_tri(feature, labels, args.norm)[0]

                loss = loss_id + loss_tri + args.kl * loss_kl
        else:
            if not args.two:
                feature, output = model(torch.cat((color, infrare), 0), cam_label=cams)
            else:
                feature, output = model(color, infrare, cam_label=cams)

            if isinstance(output, list):
                loss_id = [criterion_id(score, labels) for score in output[1:]]
                loss_id = sum(loss_id) / len(loss_id)
                loss_id = 0.5 * loss_id + 0.5 * criterion_id(output[0], labels)

                if args.tri:
                    n = 3
                else:
                    n = 2

                loss_kl = []
                for out in output[1:]:
                    output1 = out.narrow(0, 0, out.shape[0] // n)
                    output2 = out.narrow(0, (n - 1) * out.shape[0] // n, out.shape[0] // n)
                    loss_kl.append(criterion_kl(output1, output2))

                output1 = output[0].narrow(0, 0, output[0].shape[0] // n)
                output2 = output[0].narrow(0, (n - 1) * output[0].shape[0] // n, output[0].shape[0] // n)
                loss_kl = sum(loss_kl) / len(loss_kl)
                loss_kl = 0.5 * loss_kl + 0.5 * criterion_kl(output1, output2)

            else:
                loss_id = criterion_id(output, labels)

                if args.tri:
                    n = 3
                else:
                    n = 2
                output1 = output.narrow(0, 0, output.shape[0] // n)
                output2 = output.narrow(0, (n - 1) * output.shape[0] // n, output.shape[0] // n)
                loss_kl = criterion_kl(output1, output2)

            if isinstance(feature, list):
                loss_tri = [criterion_tri(feat, labels, args.norm)[0] for feat in feature[1:]]
                loss_tri = sum(loss_tri) / len(loss_tri)
                loss_tri = 0.5 * loss_tri + 0.5 * criterion_tri(feature[0], labels)[0]
            else:
                loss_tri = criterion_tri(feature, labels, args.norm)[0]

            loss = loss_id + loss_tri + args.kl * loss_kl

        if isinstance(output, list):
            _, predicted = output[0].max(1)
        else:
            _, predicted = output.max(1)
        if args.mixup or args.cutmix:
            labels = torch.max(labels, 1)[1]
            correct += predicted.eq(labels).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if math.isnan(loss.item()):
            raise ValueError('loss is Nan')

        metric_logger.meters['lr'].update(optimizer.param_groups[-1]["lr"])
        metric_logger.meters['total_loss'].update(loss.item())
        metric_logger.meters['loss_id'].update(loss_id.item())
        metric_logger.meters['loss_tri'].update(loss_tri.item())
        metric_logger.meters['loss_kl'].update(loss_kl.item())
        metric_logger.meters['acc'].update(correct / len(labels) * 100.0)
        metric_logger.meters['img/s'].update(len(labels) / (time.time() - start))

    return metric_logger.total_loss.global_avg, metric_logger.loss_id.global_avg, \
        metric_logger.loss_tri.global_avg, metric_logger.acc.global_avg


def log(logger, cmc, mAP, mINP, cmc_att, mAP_att, mINP_att):
    logger.info(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    logger.info(
        'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))


def evaluate(model, gall_loader, query_loader, query_label, gall_label, query_cam, gall_cam,
             ngall, nquery, device, args, logger):
    model.eval()
    logger.info("Extracting Gallery Feature")
    start = time.time()
    ptr = 0
    gall_feature = np.zeros((ngall, args.dim))
    gall_feature_att = np.zeros((ngall, args.dim))
    with torch.no_grad():
        for image, label, cam in gall_loader:
            batch = image.shape[0]
            image = image.to(device)
            if args.two:
                feature, feature_att = model(image, image, modal=args.test_mode[0], cam_label=cam)
            else:
                feature, feature_att = model(image, cam_label=cam)
            gall_feature[ptr:ptr + batch, :] = feature.detach().cpu().numpy()
            gall_feature_att[ptr:ptr + batch, :] = feature_att.detach().cpu().numpy()
            ptr += batch
    logger.info("Extracting Time:\t {}".format(datetime.timedelta(seconds=int(time.time() - start))))

    logger.info("Extracting Query Feature")
    start = time.time()
    ptr = 0
    query_feature = np.zeros((nquery, args.dim))
    query_feature_att = np.zeros((nquery, args.dim))
    with torch.no_grad():
        for image, label, cam in query_loader:
            batch = image.shape[0]
            image = image.to(device)
            if args.two:
                feature, feature_att = model(image, image, modal=args.test_mode[1], cam_label=cam)
            else:
                feature, feature_att = model(image, cam_label=cam)
            query_feature[ptr:ptr + batch, :] = feature.detach().cpu().numpy()
            query_feature_att[ptr:ptr + batch, :] = feature_att.detach().cpu().numpy()
            ptr += batch
    logger.info("Extracting Time:\t {}".format(datetime.timedelta(seconds=int(time.time() - start))))

    logger.info("Evaluating")
    start = time.time()
    distmat = np.matmul(query_feature, np.transpose(gall_feature))
    distmat_att = np.matmul(query_feature_att, np.transpose(gall_feature_att))

    if args.dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif args.dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    else:
        raise Exception(f"Invaild dataset: {args.dataset}")

    logger.info("Evaluation Time:\t {}".format(datetime.timedelta(seconds=int(time.time() - start))))

    log(logger, cmc, mAP, mINP, cmc_att, mAP_att, mINP_att)

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


def load_data(args, logger):
    start = time.time()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        normalize])

    transform_train_list = [
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]

    if args.color:
        transform_train_list = transform_train_list + [transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)]

    if args.re > 0:
        if args.cre:
            logger.info("Using CRE")
            transform_train_list = transform_train_list + [transform.ChannelRandomErasing(args.re, mean=[0.4814, 0.4822, 0.4465])]
        else:
            logger.info("Using RE")
            transform_train_list = transform_train_list + [transforms.RandomErasing(args.re)]

    if args.ga > 0:
        transform_train_list = transform_train_list + [transform.ChannelAdapGray(args.ga)]

    transform_color1 = deepcopy(transform_train_list)
    transform_color2 = deepcopy(transform_train_list)
    transform_infrare = deepcopy(transform_train_list)

    if args.augc:
        if args.tri:
            transform_color2 = transform_color2 + [transform.ChannelAugument()]
        else:
            transform_color1 = transform_color1 + [transform.ChannelAugument(3)]

    transform_color1 = transforms.Compose(transform_color1 + [normalize])
    transform_color2 = transforms.Compose(transform_color2 + [normalize])
    transform_infrare = transforms.Compose(transform_infrare + [normalize])

    query_cam = None
    gall_cam = None

    if args.dataset == 'sysu':
        train_set = data.SYSUData(args.data_path, img_size=(args.width, args.height),
                                  transform_color1=transform_color1,
                                  transform_color2=transform_color2,
                                  transform_thermal=transform_infrare)

        color_pos, thermal_pos = data.GenIdx(train_set.train_color_label, train_set.train_thermal_label)

        query_img, query_label, query_cam = data.process_query_sysu(args.data_path)
        gall_img, gall_label, gall_cam = data.process_gallery_sysu(args.data_path, mode=args.mode)
        query_cam = query_cam + 1
        gall_cam = gall_cam + 1

    elif args.dataset == 'regdb':
        train_set = data.RegDBData(args.data_path, args.trial, img_size=(args.width, args.height),
                                   transform_color1=transform_color1,
                                   transform_color2=transform_color2,
                                   transform_thermal=transform_infrare)

        color_pos, thermal_pos = data.GenIdx(train_set.train_color_label, train_set.train_thermal_label)
        query_cam = gall_cam = None

        # test_mode stores [gallery stem, query stem].
        if args.test_mode == [2, 1]:
            query_img, query_label = data.process_test_regdb(args.data_path, trial=args.trial, modal='visible')
            gall_img, gall_label = data.process_test_regdb(args.data_path, trial=args.trial, modal='thermal')
        elif args.test_mode == [1, 2]:
            query_img, query_label = data.process_test_regdb(args.data_path, trial=args.trial, modal='thermal')
            gall_img, gall_label = data.process_test_regdb(args.data_path, trial=args.trial, modal='visible')
        else:
            raise ValueError('RegDB --test-mode must be "1 2" or "2 1"')
    else:
        raise Exception(f"Invaild dataset: {args.dataset}")

    gallset = data.TestData(gall_img, gall_label, gall_cam,
                            transform=transform_test, img_size=(args.width, args.height))
    queryset = data.TestData(query_img, query_label, query_cam,
                             transform=transform_test, img_size=(args.width, args.height))

    # testing data loader
    gall_loader = DataLoader(gallset, batch_size=args.test_batch,
                             num_workers=args.workers, pin_memory=args.pin)
    query_loader = DataLoader(queryset, batch_size=args.test_batch,
                              num_workers=args.workers, pin_memory=args.pin)

    n_class = len(np.unique(train_set.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    logger.info('Dataset {} statistics:'.format(args.dataset))
    logger.info('  ------------------------------')
    logger.info('  subset   | # ids | # images')
    logger.info('  ------------------------------')
    logger.info('  visible  | {:5d} | {:8d}'.format(n_class, len(train_set.train_color_label)))
    logger.info('  thermal  | {:5d} | {:8d}'.format(n_class, len(train_set.train_thermal_label)))
    logger.info('  ------------------------------')
    logger.info('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    logger.info('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    logger.info('  ------------------------------')
    logger.info('Data Loading Time:\t {:.3f}'.format(time.time() - start))

    return train_set, query_loader, gall_loader, query_label, gall_label, \
        n_class, nquery, ngall, color_pos, thermal_pos, query_cam, gall_cam


def main(args):
    global checkpoint
    logger, output_dir = init_logging(args)

    if not args.random:
        fix_state(args.seed)
    else:
        logger.info('-----Starting-----')

    if args.local:
        args.dim = args.dim * 5

    logger.info(args)

    train_tb_writer = None
    te_tb_writer = None
    max_mAP = 0
    max_epoch = 0
    device = torch.device(args.device)

    train_set, query_loader, gall_loader, query_label, gall_label, \
        n_class, nquery, ngall, color_pos, thermal_pos, query_cam, gall_cam = \
        load_data(args, logger)

    if args.model == 'transreid':
        model = transreid.make_model(args, n_class)
    else:
        raise NotImplementedError('{} not implemented'.format(args.model))

    if args.checkpoint_stages is not None:
        for stage, layer in enumerate(model.base.layers):
            layer.use_checkpoint = stage in args.checkpoint_stages

    model.to(device)
    print(model)

    version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])

    logger.info('Preparing loss function')
    if version >= (1, 10):
        logger.info(f'Using label smoothing: {args.ls}')
        criterion_id = nn.CrossEntropyLoss(label_smoothing=args.ls)
    else:
        logger.info(f'torch < 0.10.0 (without label smoothing)')
        criterion_id = nn.CrossEntropyLoss()

    criterion_tri = TripletLoss_ADP(square=args.square)
    criterion_kl = KLDivLoss(args.T)

    logger.info('Preparing optimizer')

    params_group = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "bias" in key:
            if args.bias:
                lr = lr * 2
                weight_decay = 1e-4

        if "bottleneck" in key or "classifier" in key or "sie_embed" in key:
            pass
        else:
            lr /= 10

        if args.same:
            lr = args.lr

        logger.info(f"{key}: lr  {lr} \t weight_decay  {weight_decay}")

        params_group += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.adam:
        logger.info('Using Adam')
        optimizer = torch.optim.Adam(
            params_group,
            betas=(args.beta1, args.beta2)
        )
    elif args.adamw:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(
            params_group,
            betas=(args.beta1, args.beta2))
    else:
        logger.info('Using SGD')
        optimizer = torch.optim.SGD(
            params_group,
            momentum=args.momentum,
            nesterov=args.nesterov
        )

    if args.amp:
        logger.info('Mix-Precision training')
        scaler = amp.GradScaler()
    else:
        scaler = None

    logging.info('Preparing learning rate scheduler')
    if not args.cos:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=args.lr_gamma, milestones=args.milestone
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup - args.start_epoch
        )

    if args.warmup > 0:
        logger.info(f'Warm up: {args.warmup + 1}')
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.lr_warmup_decay, total_iters=args.warmup
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[args.warmup]
        )

    if args.resume:
        logging.info('Resuming')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_mAP = checkpoint.get('max_mAP', 0)
        max_epoch = checkpoint.get('max_epoch', -1)
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        rng_state = checkpoint.get('rng_state')
        if rng_state is not None:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if device.type == 'cuda' and rng_state['cuda'] is not None:
                torch.cuda.set_rng_state(rng_state['cuda'], device)

    if args.finetune:
        logging.info('Finetuning')
        checkpoint = torch.load(args.finetune, map_location='cpu')
        del checkpoint['model']['classifier.weight']
        model.load_state_dict(checkpoint['model'], strict=False)

    if args.test_only:
        evaluate(model, gall_loader, query_loader, query_label, gall_label, query_cam, gall_cam,
                 ngall, nquery, device, args, logger)
        return

    if args.tb:
        vis_path = output_dir + '_logs'
        logger.info('Preparing tensorboard logging')
        purge_step = args.start_epoch
        train_tb_writer = SummaryWriter(os.path.join(vis_path, 'train'),
                                        purge_step=purge_step)
        te_tb_writer = SummaryWriter(os.path.join(vis_path, 'test'),
                                     purge_step=purge_step)

        with open(os.path.join(vis_path, 'args.txt'), 'w', encoding='utf-8') as fp:
            fp.write(str(args))

        logger.info(f'Purge_step={args.start_epoch}')

    logger.info('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # redefine sampler at each epoch
        sampler = data.IdentitySampler(train_set.train_color_label, train_set.train_thermal_label,
                                       color_pos, thermal_pos, args.num_pos, args.batch_size)

        train_set.cIndex = sampler.index1  # color index
        train_set.tIndex = sampler.index2  # thermal index

        loader_batch = args.batch_size * args.num_pos

        mixup_transforms = []

        if args.mixup or args.cutmix:
            if args.mixup:
                print('Using MixUp')
                mixup_transforms.append(transform.RandomMixup(n_class, p=args.mixup_prob,
                                        alpha=args.mixup_alpha))
            if args.cutmix:
                print("Using Cutmix")
                mixup_transforms.append(transform.RandomCutmix(n_class, p=args.cutmix_prob,
                                                               alpha=args.cutmix_alpha))
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
        else:
            fn = None

        train_loader = DataLoader(train_set, batch_size=loader_batch, pin_memory=args.pin, collate_fn=fn,
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

        train_loss, train_loss_id, train_loss_tri, train_acc = train_one_epoch(
            model, criterion_id, criterion_tri, optimizer, train_loader, device,
            epoch, args, scaler, logger, criterion_kl)

        if train_tb_writer is not None:
            train_tb_writer.add_scalar('Train/train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('Train/train_loss_id', train_loss_id, epoch)
            train_tb_writer.add_scalar('Train/train_loss_tri', train_loss_tri, epoch)
            train_tb_writer.add_scalar('Train/train_acc', train_acc, epoch)

        lr_scheduler.step()

        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = evaluate(model, gall_loader, query_loader,
                                                              query_label, gall_label, query_cam,
                                                              gall_cam, ngall, nquery, device, args,
                                                              logger)

        if te_tb_writer is not None:
            te_tb_writer.add_scalar('Test/Rank1', cmc_att[0], epoch)
            te_tb_writer.add_scalar('Test/Rank5', cmc_att[4], epoch)
            te_tb_writer.add_scalar('Test/Rank10', cmc_att[9], epoch)
            te_tb_writer.add_scalar('Test/Rank20', cmc_att[19], epoch)
            te_tb_writer.add_scalar('Test/mAP', mAP_att, epoch)
            te_tb_writer.add_scalar('Test/mINP', mINP_att, epoch)

        if output_dir:
            is_best = mAP_att > max_mAP
            if is_best:
                max_epoch = epoch
                max_mAP = mAP_att

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_mAP': max_mAP,
                'max_epoch': max_epoch,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'rng_state': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                },
            }

            if is_best:
                best_path = os.path.join(output_dir, 'checkpoint_max_mAP.pth')
                torch.save(checkpoint, best_path + '.tmp')
                os.replace(best_path + '.tmp', best_path)

            last_path = os.path.join(output_dir, 'checkpoint_last.pth')
            torch.save(checkpoint, last_path + '.tmp')
            os.replace(last_path + '.tmp', last_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        logger.info('Training time: {}, max_mAP: {}, max epoch: [{}]'.format(total_time_str, max_mAP, max_epoch))


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Cross-Modality Person Reidentification')

    # parameters for model configuration
    parser.add_argument('--width', default=144, type=int, help='input image width')
    parser.add_argument('--height', default=288, type=int, help='input image height')
    parser.add_argument('--model', default='transreid', help='model variant')
    parser.add_argument('--siecoe', default=3.0, type=float, help='coefficience of SIE')
    parser.add_argument('--droppath', default=0.1, type=float, help='drop path rate')
    parser.add_argument('--activation-checkpointing', action='store_true',
                        help='recompute SwinV2 block activations during training to reduce GPU memory')
    parser.add_argument('--checkpoint-stages', nargs='+', type=int, choices=range(4),
                        default=None, help='recompute only these SwinV2 stages (0-3); overrides --activation-checkpointing')
    parser.add_argument('--rearrange', action='store_true', help='rearrange tokens')
    parser.add_argument('--nonlinear', action='store_true', help='nonlinear classification head')
    parser.add_argument('--local', action='store_true', help='local feature learning')
    parser.add_argument('--same', action='store_true', help='use the same lr for the whole network')
    parser.add_argument('--backbone', default='vit_base_patch16_224_TransReID', help='model backbone')
    # parser.add_argument('--pretrain', default='./swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth')
    parser.add_argument('--pretrain', default='./swinv2_base_patch4_window12_192_22k.pth',
                        help='path to pretrained backbone')
    parser.add_argument('--stride', default=2, type=int, help='number of stage for modality-specific module')
    parser.add_argument('--camera-num', default=0, type=int, help='number of cameras')
    parser.add_argument('--view-num', default=0, type=int, help='number of viewports')
    parser.add_argument('--shuffle', default=2, type=int, help='number of shuffle group')
    parser.add_argument('--shift', default=5, type=int, help='number of shift num')
    parser.add_argument('--divide', default=4, type=int, help='number of devide length')

    # parameters for training configuration
    parser.add_argument('--dataset', default='sysu', choices=['sysu', 'regdb'],
                        help='dataset name: regdb or sysu')
    parser.add_argument('--data-path', default='../dataset/SYSU-MM01', help='path to dataset')
    parser.add_argument('--device', default='cuda:0', help='device: cuda:x or cpu')
    parser.add_argument('--batch-size', default=8, type=int, help='training batch size')
    parser.add_argument('--test-batch', default=64, type=int, help='testing batch size')
    parser.add_argument('--num-pos', default=4, type=int, help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor search (only for SYSU dataset)')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--ls', default=0, type=float, help='label smoothing (torch >= 0.10.0)')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--augc', action='store_true', help='use channel aug or not')
    parser.add_argument('--color', action='store_true', help='use channel aug or not')
    parser.add_argument('--cre', action='store_true', help='use channel random erase or random erase')
    parser.add_argument('--bias', action='store_true', help='the lr of bias is doubled')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp Data Augumentation')
    parser.add_argument('--mixup-alpha', default=1.0, type=float, help='hyper parameter for MixUp')
    parser.add_argument('--mixup-prob', default=0.5, type=float, help='hyper parameter for MixUp')
    parser.add_argument('--cutmix', action='store_true', help='Use Cutmix Data Augumentation')
    parser.add_argument('--cutmix-alpha', default=1.0, type=float, help='hyper parameter for Cutmix')
    parser.add_argument('--cutmix-prob', default=0.5, type=float, help='hyper parameter for Cutmix')
    parser.add_argument('--kl', default=0, type=float, help='hyperparameter for kl loss')
    parser.add_argument('--T', default=3, type=float, help='temperature for kl loss')
    parser.add_argument('--two', action='store_true', help='multi modality training')
    parser.add_argument('--tri', action='store_true', help='tri modality training')
    parser.add_argument('--norm', action='store_true', help='normalize feature')
    # !!!usually 0.25 AGW adopts 0.5 and do not adopt channel_re
    parser.add_argument('--re', default=0.5, type=float, help='the probability for random erase')
    parser.add_argument('--ga', default=0, type=float, help='the probability for gray scale aug')
    parser.add_argument('--test-mode', nargs='+', type=int, default=[1, 2],
                        help='[gallery stem, query stem]: 1=visible, 2=infrared')
    parser.add_argument('--square', default=0, type=int, help='Whether to use square difference')
    parser.add_argument('--dim', default=1024, type=int, help='Final dimension of the output feature')

    # parameters for optimizer
    parser.add_argument('--adam', action='store_true',
                        help='using adam as optimizer (default: sgd)')
    parser.add_argument('--adamw', action='store_true',
                        help='using adamw as optimizer (default: sgd)')
    parser.add_argument('--nesterov', action='store_true',
                        help='using nesterov for sgd')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for Adam')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='weight decay')

    # parameters for learning rate scheduler
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='factor for decaying the learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[40, 70],
                        help='milestone to decay the learning rate by lr_gamma')
    parser.add_argument("--lr-warmup-decay", default=0.1, type=float,
                        help="the init ratio of learning rate during the first epoch")
    parser.add_argument('--warmup', default=10, type=int, help='warm up epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='use CosineAnnealing learning rate scheduler')

    # parameters for Auxiliary config
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', default='', help='resume from checkpoint')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path to output log and model')
    parser.add_argument('--output-log', default='log.txt', help='filename of the output log')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument("--test-only", help="Only test the model", action="store_true", )
    parser.add_argument('--amp', action='store_true', help='Use AMP training')
    parser.add_argument('--random', action='store_true', help='do not fix random seed')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--pin', action='store_true',
                        help='Whether to pin memory')

    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
