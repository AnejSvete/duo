import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _is_looping_model(algo_name, looping_type):
    """Determine if model is a looping model based on algorithm and looping type."""
    # LT with non-constant looping is considered a looping model
    if algo_name == "lt" and looping_type != "constant":
        return True
    # AR uses constant looping (single-pass) so it's not a looping model
    # MDLM, D3PM, SEDD are also non-looping
    return False


omegaconf.OmegaConf.register_new_resolver(
    "adaptive_batch_size",
    lambda algo_name, looping_type: (
        256 if _is_looping_model(algo_name, looping_type) else 1024
    ),
)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=128):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device
            )
            all_samples.extend(list(text_samples))
    generative_ppl = 0.0
    entropy = 0.0
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print("Generative perplexity:", generative_ppl)
        print("Sample entropy:", entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as f:
        json.dump(
            {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "generated_seqs": all_samples,
            },
            f,
            indent=4,
        )
    print("Samples saved at:", samples_path)


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Perplexity Eval.")

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds, test_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds, test_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path, tokenizer=tokenizer, config=config
        )
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

    # Final evaluation on test dataset
    if test_ds is not None:
        logger.info("Running final evaluation on test dataset.")
        trainer.test(model, test_ds)


def _prepare_data(config, logger, tokenizer):
    """Prepare and cache data without training."""
    logger.info("Starting Data Preparation.")
    logger.info(
        f"Preparing data for: {config.data.language}, format: {config.data.properties.format}"
    )

    # Just load the dataloaders - this triggers data generation and caching
    train_ds, valid_ds, test_ds = dataloader.get_dataloaders(config, tokenizer)

    # Print info about what was generated
    logger.info(f"Train dataset size: {len(train_ds.dataset)}")
    logger.info(f"Validation dataset size: {len(valid_ds.dataset)}")
    if test_ds is not None:
        logger.info(f"Test dataset size: {len(test_ds.dataset)}")

    # Print 50 example sequences from the training set
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE PREPARED EXAMPLES (first 50 from training set)")
    logger.info("=" * 80)

    num_examples_to_print = min(50, len(train_ds.dataset))
    for i in range(num_examples_to_print):
        example = train_ds.dataset[i]

        # Decode the tokenized sequence back to text
        if hasattr(example, "input_ids"):
            tokens = example.input_ids
        elif isinstance(example, dict) and "input_ids" in example:
            tokens = example["input_ids"]
        else:
            tokens = example

        # Convert token IDs back to text
        if hasattr(tokenizer, "decode"):
            text = tokenizer.decode(tokens)
        else:
            # Fallback: try to convert tokens directly
            text = " ".join(
                [
                    tokenizer.idx_to_token.get(int(t), f"<UNK:{t}>")
                    for t in tokens
                    if int(t) != tokenizer.pad_token_id
                ]
            )

        logger.info(f"\nExample {i+1}/{num_examples_to_print}:")
        logger.info(f"  Raw tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        logger.info(f"  Decoded: {text}")

    logger.info("\n" + "=" * 80)
    logger.info("Data preparation complete. Data is now cached and ready for training.")
    logger.info("=" * 80)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    # Handle data preparation mode separately (doesn't need model or full config)
    if config.mode == "prepare_data":
        logger.info("Mode: Data Preparation Only")
        logger.info(f"Language: {config.data.language}")
        logger.info(f"Format: {config.data.properties.format}")
        logger.info(f"Model Length: {config.model.length}")
        _prepare_data(config, logger, tokenizer)
        return

    # For training modes, print full config
    _print_config(config, resolve=True, save_cfg=True)

    # Initialize model for other modes
    if config.algo.name == "ar":
        diffusion_model = algo.AR
    elif config.algo.name == "lt":
        diffusion_model = algo.LT
    elif config.algo.name == "mdlm":
        diffusion_model = algo.MDLM
    elif config.algo.name == "d3pm":
        diffusion_model = algo.D3PMAbsorb
    elif config.algo.name == "sedd":
        diffusion_model = algo.SEDDAbsorb
    else:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    kwargs = {
        "diffusion_model": diffusion_model,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    if config.mode == "sample_eval":
        _generate_samples(**kwargs)
    elif config.mode == "ppl_eval":
        _eval_ppl(**kwargs)
    else:
        _train(**kwargs)


if __name__ == "__main__":
    main()
