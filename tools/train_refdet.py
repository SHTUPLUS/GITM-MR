#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import time
import numpy as np
import pandas as pd
import random
import json
import sys
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    RefdetEvaluator,
    verify_results,
)
from detectron2.data import (
    build_refdet_train_loader,
    build_refdet_test_loader
)
from detectron2.modeling import GeneralizedRCNNWithTTA

class Trainer(DefaultTrainer):
    def __init__(self, cfg, pretrain_opts):
        date = time.strftime('%y%m%d', time.localtime())
        explog_path = os.path.join(cfg.OUTPUT_DIR, '..', f'explog_{date}.csv')
        pd.DataFrame([cfg.OUTPUT_DIR, cfg.MODEL.MEMO]).to_csv(explog_path, mode='a', index=False, header=False)

        super().__init__(cfg, pretrain_opts)
        # create results csv
        csv_path = os.path.join(self.cfg.OUTPUT_DIR, 'train.csv')
        if not os.path.exists(csv_path):
            pd.DataFrame([['train_acc', 'train_ncorrect', 'train_ntotal', 'train_loss', 'lr']]).to_csv(csv_path, mode='w', index=False, header=False)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, 'inference'), exist_ok=True)

        csv_path = os.path.join(self.cfg.OUTPUT_DIR, 'inference', 'val.csv')
        if not os.path.exists(csv_path): # for resume
            head = ['match_acc', 'grd_acc', 'mrr_acc']
            pd.DataFrame([head]).to_csv(csv_path, mode='w', index=False, header=False)

        self.eval_only = cfg.eval_only
        self.max_acc = 0
        self.cfg = cfg
        self.pretrain_opts = pretrain_opts

    def build_optimizer(cls, cfg, model):
        """
                Build an optimizer from config.
                """
        from typing import Any, Dict, List, Set
        from detectron2.solver.build import maybe_add_gradient_clipping
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        separate_lr = cfg.SOLVER.SEP_LR
        freeze_backbone = cfg.SOLVER.FIX_BACKBONE
        
        def is_head_modules(key):
            allowed = cfg.MODEL.HEAD_MODULES
            for a in allowed:
                if a in key: return True
            return False
        tmp = list()
        
        for name,module in model.named_modules():
            head_param = separate_lr and is_head_modules(name)
            uniter_param = 'uniter' in name

            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad or (freeze_backbone and not is_head_modules(name)):
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                if head_param: tmp.append(key)
                if head_param: lr = cfg.SOLVER.HEAD_LR
                else: lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY

                if uniter_param:
                    lr = cfg.SOLVER.UNITER_LR
                    weight_decay = cfg.SOLVER.UNITER_WEIGHT_DECAY
                    if any(nd in key for nd in no_decay) or 'LayerNorm' in name:
                        weight_decay=0.0

                else:

                    if isinstance(module, norm_module_types):
                        weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM 

                    elif key == "bias":
                        # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                        # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                        # hyperparameters are by default exactly the same as for regular
                        # weights.
                        lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
                        weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        opt_type = cfg.SOLVER.OPTIMIZER.lower()
        if opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
            )
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(
                params, cfg.SOLVER.BASE_LR,
                betas=cfg.SOLVER.ADAM_BETAS
            )
        elif opt_type=='adamw':
            optimizer = torch.optim.AdamW(
            params, cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.ADAM_BETAS
        )
        else:
            raise NotImplementedError('only sgd, adam, adamw supported currently')
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def save_after_test(self, results):
        acc = results.get('val/retrieval_acc', None)

        if acc is not None and not self.eval_only:
            if self.max_acc<acc:
                # save
                self.max_acc = acc
                self.checkpointer.save('best', iteration=self.scheduler.last_epoch)
                print('best checkpoint acc=', acc)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_refdet_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_refdet_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = RefdetEvaluator(
            dataset_name,
            cfg,
            distributed=True,
            output_dir=output_folder
        )
        return evaluator

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def run_step(self):
        """
        Implement the standard training logic described above.
        """

        ################## Manually defined program for training #######################
        load_uniter = False
        match_loss_weight = self.cfg.MODEL.REF.MATCH_LOSS_WEIGHT

        if self.cfg.MODEL.REF.AUTO_WEIGHT:
            if self.iter <= 40000 and self.iter <= self.one_epoch_iter * 10:
                match_loss_weight = 0.0

        if self.cfg.MODEL.REF.AUTO_UNITER:
            if self.iter > 50000 or self.iter > self.one_epoch_iter * 15:
                load_uniter = True
        ################################################################################

        if load_uniter:
            self.cfg.defrost()
            logger = logging.getLogger("detectron2.trainer")
            self.checkpointer.save('scratch', iteration=self.scheduler.last_epoch)

            self.cfg.MODEL.WEIGHTS = os.path.join(self.checkpointer.save_dir, 'scratch.pth')
            self.cfg.MODEL.REF.VIS_FEAT_TYPE = 'uniter'
            self.cfg.MODEL.REF.AUTO_UNITER = False

            logger.info("Loading UNITER and continue...")
            super().__init__(self.cfg, self.pretrain_opts)
            self.resume_or_load(resume=False)
            self.cfg.freeze()

        assert self.model.training, "[Trainer] model was changed to eval mode!"

        data = next(self._data_loader_iter)

        loss_dict, results_dict = self.model(data)
        if self.cfg.MODEL.REF.AUTO_WEIGHT:
            loss_dict["match_loss"] /= self.cfg.MODEL.REF.MATCH_LOSS_WEIGHT
            loss_dict["match_loss"] *= match_loss_weight

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        if torch.isfinite(losses).all():
            losses.backward()
        else:
            logger = logging.getLogger("detectron2.trainer")
            logger.info("Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                ))
            return

        ncorrect, ntotal = [results_dict[k] for k in ['ncorrect', 'ntotal']]

        if 'num_matchcorrect' in results_dict:
            self.predict_match = True
            # true nc: true positive
            # false nc: true negative
            nmcorrect, nmtotal, tnc, fnc, tnt, fnt = [results_dict[k] for k in ['num_matchcorrect', 'num_all',
                                                                             'true_nc', 'false_nc',
                                                                             'true_nt','false_nt']]

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = {'train/{}'.format(k):v for k,v in loss_dict.items()}
            metrics_dict.update({'train/'+k:v for k,v in results_dict.items() if 'reward' in k})
            metrics_dict.update({'train/retr_nc': ncorrect,
                                 'train/retr_nt': ntotal})
            if self.predict_match:
                metrics_dict['train/match_nc'] = nmcorrect
                metrics_dict['train/match_nt'] = nmtotal
                metrics_dict['train/true_nc'] = tnc
                metrics_dict['train/false_nc'] = fnc
                metrics_dict['train/true_nt'] = tnt
                metrics_dict['train/false_nt'] = fnt

            self._write_metrics(metrics_dict) # can auto convert to numpy from tensor
            self._detect_anomaly(losses, loss_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        self.optimizer.step()
        torch.cuda.empty_cache()

    def _write_metrics(self, metrics_dict_across_machines: dict):
        """
        Args:
            metrics_dict_across_machines (dict): dict of scalar metrics
        """
        metrics_dict_across_machines = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict_across_machines.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.

        # print('check backend before gathering: ', torch.distributed.get_backend())
        all_metrics_dict = comm.gather(metrics_dict_across_machines)
        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            loss_keys = [k for k in all_metrics_dict[0] if 'loss' in k or 'reward' in k]
            sum_keys = ['train/retr_nc','train/retr_nt']
            if self.predict_match:
                sum_keys += ['train/match_nc','train/match_nt', 'train/true_nc', 'train/true_nt',
                             'train/false_nc', 'train/false_nt']

            metrics_dict_across_machines ={k:np.mean([x[k] for x in all_metrics_dict]) for k in loss_keys}
            metrics_dict_across_machines.update({k:np.sum([x[k] for x in all_metrics_dict]) for k in sum_keys})
            matchonly_correct = metrics_dict_across_machines['train/retr_nc']
            matchonly_total = metrics_dict_across_machines['train/retr_nt']
            matchonly_acc = round(matchonly_correct / float(matchonly_total), 4)
            metrics_dict_across_machines['train/retr_acc'] = matchonly_acc
            if self.predict_match:
                match_correct = metrics_dict_across_machines['train/match_nc']
                match_total = metrics_dict_across_machines['train/match_nt']
                true_nc, true_nt, false_nc, false_nt = [metrics_dict_across_machines[k] for k in
                                                        ['train/true_nc','train/true_nt','train/false_nc', 'train/false_nt']]

                match_acc = round(match_correct / float(match_total), 4)
                true_recall = round(true_nc / float(true_nt), 4)
                false_recall = round(false_nc / float(false_nt), 4)

                metrics_dict_across_machines['train/match_acc'] = match_acc
                metrics_dict_across_machines['train/true_recall'] = true_recall
                metrics_dict_across_machines['train/false_recall'] = false_recall

            lr = self.optimizer.param_groups[-1]["lr"]
            output_dir = self.cfg.OUTPUT_DIR
            csv_path = os.path.join(output_dir, 'train.csv')
            # need to convert numpy myself for loss
            write_csv = [lr, matchonly_acc, matchonly_correct, matchonly_total]
            if self.predict_match:
                write_csv.append(match_acc)
            write_csv += [metrics_dict_across_machines[k] for k in loss_keys]
            pd.DataFrame([write_csv]).to_csv(csv_path, index=False, header=False, mode='a')

            if len(metrics_dict_across_machines) > 1:
                self.storage.put_scalars(**metrics_dict_across_machines)

    def build_hooks(self):
        ret = super(Trainer, self).build_hooks()
        train_loader_period = self.cfg.DATALOADER.TRAIN_RESAMPLE_PERIOD
        if train_loader_period > 0:
            print('resample train loader every {} iterations'.format(train_loader_period))
            ret.append(hooks.PeriodicLoaderHook(train_loader_period))
        return ret

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.eval_only = args.eval_only

    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def main(args):
    pretrain_opts=args
    cfg = setup(args)
    print('setting seed')
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.eval_only:
        model = Trainer.build_model(cfg, pretrain_opts)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg,pretrain_opts)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    return trainer.train()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    # add uniter config
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config

    cfg = setup(args)
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


