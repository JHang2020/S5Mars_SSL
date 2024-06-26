import logging

from .cityscapes import build_city_semi_loader, build_cityloader
from .pascal_voc import build_voc_semi_loader, build_vocloader
from .pku2mars import build_marsloader,build_mars_semi_loader
from .ai4mars import build_ai4marsloader,build_ai4mars_semi_loader
logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "cityscapes_semi":
        train_loader_sup, train_loader_unsup = build_city_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_cityloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader
    elif cfg_dataset["type"] == "pku2mars_semi":
        train_loader_sup, train_loader_unsup = build_mars_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_marsloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader
    
    elif cfg_dataset["type"] == "ai4mars_semi":
        train_loader_sup, train_loader_unsup = build_ai4mars_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_ai4marsloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "cityscapes":
        train_loader_sup = build_cityloader("train", cfg, seed=seed)
        val_loader = build_cityloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    elif cfg_dataset["type"] == "pascal_semi":
        train_loader_sup, train_loader_unsup = build_voc_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "pascal":
        train_loader_sup = build_vocloader("train", cfg, seed=seed)
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )
