import logging
import os
import hydra
import torch

from datasets.data import NoisyCleanSet
from enh.demucs.demucs import Demucs
from enh.FullSubNet.fullsubnet import FullSubNet

from trainer import distrib
from trainer.trainer import Trainer
from trainer.executor import start_ddp_workers

logger = logging.getLogger(__name__)


def run(args):
    distrib.init(args)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    if args.model == 'Demucs':
        model = Demucs(**args.demucs, sample_rate=args.sample_rate)
    elif args.model == 'FullSubNet':
        model = FullSubNet(**args.fullsubnet, sample_rate=args.sample_rate)
    else:
        raise NotImplementedError

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}

    # Building datasets and loaders
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, 
        acoustic_path=args.acoustic_train_path, ph_logits_path=args.ph_logits_train_path, **kwargs)
    print("Total number of train files: %s" % len(tr_dataset))
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(args.dset.valid,
        acoustic_path=args.acoustic_valid_path, ph_logits_path=args.ph_logits_valid_path, **kwargs)
        print("Total number of valid files: %s" % len(cv_dataset))
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(args.dset.test,
        acoustic_path=args.acoustic_test_path, ph_logits_path=args.ph_logits_test_path, **kwargs)
        print("Total number of test files: %s" % len(tt_dataset))
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
        print("learn rate: ", args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        print("learn rate: ", args.lr)
    
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct trainer
    trainer = Trainer(data, model, optimizer, args)
    trainer.train()


def _main(args):
    global __file__
    # Updating paths in config
    if hydra.__version__ >= '1.0.0':
        args.update(args.finetune)
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    logging.info(args)
    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)

if hydra.__version__ >= '1.0.0':
    kwargs = {"config_path": "conf/", "config_name": "base_1.0.0.yaml"}
else:
    kwargs = {"config_path": "conf/base.yaml"}

@hydra.main(**kwargs)
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
