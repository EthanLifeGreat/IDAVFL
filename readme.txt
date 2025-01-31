Algorithm:
    encoder_matching: main algorithm for dvfl
    encoder_matching_baseline: define baseline algorithms
    encoder_matching_compare: comparing baseline algorithms and ours
    encoder_matching_incremental*: dynamic algorithms
    encoder_matching_multi_passive: define multi-passive-party algorithm
Data:
    4 Data-sets: BCW, DCC, EPS5k, HAR
Model:
    model checkpoints and model class files
Record:
    running results
Test:
    corr_heatmap: plot correlation heatmap
    homo_float: demo of homomorphic encryption with support of float data type
    incremental_test*: RUN test of dynamic algorithms
    multi_passive_compare: RUN multi-passive-party algorithm
    plot_continuous_incremental: plot figures of dynamic algorithms
    plot_incremental_time: plot figs
    plot_multi_passive: plot figs
    sklearn_score: sklearn baselines
    split_csv*: split csv files for FATE to run
Utils:
    CV: cross_validation
    utils: utils
args: define algorithm parameters for different data-sets
main: RUN non-dynamic comparing algorithms

