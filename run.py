from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from pyrootutils import find_root
import os
from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rl4co import utils
from rl4co.utils import RL4COTrainer

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def run(cfg: DictConfig) -> Tuple[dict, dict]:

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    log.info(f"Instantiating model <{cfg.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model, env)

    log.info("Instantiating callbacks...") 

    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...") 
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"), model=model)

    log.info("Instantiating trainer...") 
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"): 
        log.info("Starting training!")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path")) 

        train_metrics = trainer.callback_metrics

    if cfg.get("test"): 
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics 

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def train(cfg: DictConfig) -> Optional[float]:

    utils.extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, 
        metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":

    train()
