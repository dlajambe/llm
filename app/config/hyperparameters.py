class Hyperparams:
    """Stores hyperparameters used during model construction, training
    and evaluation.
    """
    block_size = 256
    n_embed = 384
    n_heads = 6
    ff_proj_factor = 4
    if n_embed % n_heads != 0:
        raise ValueError(
            'Embedding dimension must be divisible by number of heads')
    head_size = int(n_embed / n_heads)
    n_trans_blocks = 6
    batch_size = 64
    lr = 3e-4
    max_training_iters = 100000
    seed = 1337
    eval_interval = 500
    eval_batches = 200
    dropout_frac = 0.35