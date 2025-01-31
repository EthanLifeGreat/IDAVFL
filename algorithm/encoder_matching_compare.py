from algorithm.encoder_matching_baseline import *


def encoder_matching_compare(arguments):
    if arguments.test_using_train_data:
        train_dataset, _ = utils.datasets(arguments)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=len(train_dataset),
                                                  shuffle=False)
    else:
        train_dataset, test_dataset = utils.datasets(arguments)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=arguments.train_loader_shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset),
                                                  shuffle=False)

    encoder_matching_train_baseline(train_loader, arguments)
    baseline = encoder_matching_test_baseline(test_loader, arguments)

    encoder_matching_train_baseline_sae(train_loader, arguments)
    baseline_sae = encoder_matching_test_baseline_sae(test_loader, arguments)

    encoder_matching_train(train_loader, arguments)
    ours = encoder_matching_test(test_loader, arguments)

    encoder_matching_train_baseline_sae_high(train_loader, arguments)
    baseline_sae_high = encoder_matching_test_baseline_sae_high(test_loader, arguments)

    encoder_matching_train_baseline_high(train_loader, arguments)
    baseline_high = encoder_matching_test_baseline_high(test_loader, arguments)

    print("\nBaseline:\t{}\nBaseline (SAE):\t{}\nOur model:\t{}\nBaseline (SAE HIGH):\t{}\nBaseline (High):\t{}\n"
          .format(baseline, baseline_sae, ours, baseline_sae_high, baseline_high))


def encoder_matching_train_compare(train_loader, arguments):
    encoder_matching_train_baseline(train_loader, arguments)
    encoder_matching_train_baseline_sae(train_loader, arguments)
    encoder_matching_train(train_loader, arguments)
    encoder_matching_train_baseline_sae_high(train_loader, arguments)
    encoder_matching_train_baseline_high(train_loader, arguments)


def encoder_matching_test_compare(test_loader, arguments):
    baseline = encoder_matching_test_baseline(test_loader, arguments)
    baseline_sae = encoder_matching_test_baseline_sae(test_loader, arguments)
    ours = encoder_matching_test(test_loader, arguments)
    baseline_sae_high = encoder_matching_test_baseline_sae_high(test_loader, arguments)
    baseline_high = encoder_matching_test_baseline_high(test_loader, arguments)

    return np.stack([baseline, baseline_sae, ours, baseline_sae_high, baseline_high])
