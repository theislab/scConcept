import os
import sys
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only

from lamin_dataloader import GeneIdTokenizer
from concept.data import AnnDataModule
from concept import ContrastiveModel
import wandb
from lightning.pytorch.strategies import DDPStrategy
from hydra import compose, initialize


def train(cfg: DictConfig):
    """
    Train a model using the configuration in cfg.

    Args:
        cfg: Configuration dictionary.
    """
    if 'val' in cfg.datamodule.dataset and cfg.datamodule.dataset.val is not None:
        val_loader_names = sorted(list(cfg.datamodule.dataset.val.keys()))
    else:
        val_loader_names = []
        
    gene_mapping = pd.read_pickle(cfg.PATH.gene_mapping_path).to_dict()
    
    split = {}
    for key, filenames in cfg.PATH.SPLIT.items():
        if filenames is not None:
            split[key] = [os.path.join(cfg.PATH.ADATA_PATH, file) for file in filenames]
    
    datamodule_args = {    
        'split': split,
        'panels_path': cfg.PATH.PANELS_PATH,
        'columns': cfg.datamodule.columns,
        'precomp_embs_key': cfg.datamodule.precomp_embs_key,
        'normalization': cfg.datamodule.normalization,
        'gene_sampling_strategy': cfg.datamodule.gene_sampling_strategy,
        'model_speed_sanity_check': cfg.datamodule.model_speed_sanity_check,
        # make sure to pass a copy to avoid being modified before uploading to wandb:
        'dataset_kwargs': {**OmegaConf.to_container(cfg.datamodule.dataset, resolve=True, throw_on_missing=True)}, 
        'dataloader_kwargs': {**OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)},
        'val_loader_names': val_loader_names,
        'tokenizer': GeneIdTokenizer(gene_mapping)
    }
    datamodule = AnnDataModule(**datamodule_args)

    if cfg.wandb.enabled:
        if cfg.wandb.entity is None or cfg.wandb.project is None or cfg.wandb.run_name is None:
            raise ValueError("wandb.entity, wandb.project, and wandb.run_name are required when wandb.enabled is True")
        logger = WandbLogger(name=cfg.wandb.run_name, entity=cfg.wandb.entity, project=cfg.wandb.project, save_dir=cfg.PATH.PROJECT_PATH, log_model=False)
    
    CHECKPOINT_PATH = "dummy"
    if rank_zero_only.rank == 0:
        CHECKPOINT_PATH = os.path.join(cfg.PATH.CHECKPOINT_ROOT, logger.experiment.id if cfg.wandb.enabled else 'dummy')
        if cfg.wandb.enabled:
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    trainer_kwargs = {
        'max_steps': cfg.model.training.max_steps,
        'accelerator': cfg.model.training.accelerator,
        'devices': cfg.model.training.devices,
        'num_nodes': int(os.environ['SLURM_JOB_NUM_NODES']) if 'SLURM_JOB_NUM_NODES' in os.environ else cfg.model.training.num_nodes,
        'logger': logger if cfg.wandb.enabled else None,
        'val_check_interval': cfg.model.training.val_check_interval,
        'check_val_every_n_epoch': cfg.model.training.check_val_every_n_epoch,
        'limit_train_batches': cfg.model.training.limit_train_batches,
        'callbacks': [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(dirpath=CHECKPOINT_PATH, filename='min_train_loss', monitor='train/loss', mode='min', every_n_epochs=1, save_top_k=1),
            ModelCheckpoint(dirpath=CHECKPOINT_PATH, filename='min_val_loss', monitor='val/loss', mode='min', every_n_epochs=1, save_top_k=1),
            ModelCheckpoint(dirpath=os.path.join(CHECKPOINT_PATH, 'epochs'), filename='{epoch}', every_n_epochs=1, save_on_train_epoch_end=True, save_top_k=-1, save_last='link'),
            ModelCheckpoint(dirpath=os.path.join(CHECKPOINT_PATH, 'steps'), filename='{step}', every_n_train_steps=10000, monitor='train/loss', save_top_k=-1), # save a checkpoint every 10K steps
        ],
    }
    trainer = L.Trainer(**trainer_kwargs, 
                        strategy=DDPStrategy(find_unused_parameters=True),
                        precision='bf16-mixed', 
                        use_distributed_sampler=False,
                        accumulate_grad_batches=cfg.model.training.accumulate_grad_batches,
                        )


    model_args = {
        'config': cfg.model,
        'pad_token_id': gene_mapping['<pad>'],
        'cls_token_id': gene_mapping['<cls>'],
        'vocab_size': len(gene_mapping),
        'world_size': trainer.world_size, 
        'val_loader_names': val_loader_names, 
        'label_keys_to_monitor': cfg.datamodule.label_keys_to_monitor,
        'batch_keys_to_monitor': cfg.datamodule.batch_keys_to_monitor,
        'precomp_embs_key': cfg.datamodule.precomp_embs_key,
    }
    model = ContrastiveModel(**model_args)

    if not cfg.initialize.resume and cfg.model.training.validate_before_training:
        trainer.validate(model=model, 
                        datamodule=datamodule,
                        )
    
    if cfg.initialize.resume:
        ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, cfg.initialize.run_id, cfg.initialize.checkpoint)
        model = ContrastiveModel.load_from_checkpoint(ckpt_path, **model_args, strict=False)
    
    trainer.fit(model=model, datamodule = datamodule)


if __name__ == "__main__":
    bash_cfg = OmegaConf.from_cli()
    resume = bash_cfg.pop("initialize.resume", False)
    if resume:
        run_id = bash_cfg.pop("initialize.run_id")
        checkpoint = bash_cfg.pop("initialize.checkpoint")
        
        wandb.login()
        api = wandb.Api()
        run = api.run(f'{bash_cfg.wandb.entity}/{bash_cfg.wandb.project}/{run_id}')
        print(f"Resuming training for {run.id} ...")
        cfg = DictConfig(run.config)

        cfg = OmegaConf.merge(cfg, bash_cfg) 
        print(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
                
        # cfg.model.training.val_check_interval = float(cfg.model.training.val_check_interval + 0.1) # for a bug in pytorch-lightning
        cfg.model.training.limit_train_batches = float(cfg.model.training.limit_train_batches)
        
        cfg.initialize.resume = True
        cfg.initialize.run_id = run_id
        cfg.initialize.checkpoint = checkpoint
    else:
        print(f"Starting new training ...")
        print('overrides:', sys.argv[1:])
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name="config", overrides=sys.argv[1:])
            
    train(cfg)
