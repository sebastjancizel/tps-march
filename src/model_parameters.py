import config


LGB_PARAMS = dict(
    random_state=config.RANDOM_STATE,
    cat_l2=30.73295435314998,
    cat_smooth=80.3621043082745,
    colsample_bytree=0.8302782612074199,
    learning_rate=0.010902370728351977,
    max_bin=308,
    max_depth=74,
    metric="auc",
    min_child_samples=259,
    min_data_per_group=256,
    n_estimators=1600000,
    n_jobs=-1,
    num_leaves=247,
    reg_alpha=5.083336018044204,
    reg_lambda=5.809098429470294,
    subsample=0.6451435162410799,
    subsample_freq=1,
    verbose=-1,
    # device_type="gpu"
)

XGB_PARAMS = dict(
    random_state=config.RANDOM_STATE,
    n_estimators=10000,
    verbosity=1,
    eval_metric="auc",
    tree_method="gpu_hist",
    gpu_id=0,
    alpha=3.073140869465407,
    colsample_bytree=0.5397647352121717,
    gamma=1.0788677134823792,
    reg_lambda=6.642032329676142,
    learning_rate=0.0648950102448977,
    max_bin=589,
    max_depth=10,
    min_child_weight=2.8044867047984243,
    subsample=0.948400328902343,
)
