import sys

import numpy as np
import utils.utils as utils
from algorithm.encoder_matching import check_safety_requirements
from model.Module import *
from model.SAETrainer import sae_trainer, sae_trainer_with_alignment

criterion = nn.MSELoss()


def encoder_matching(arguments):
    if arguments.test_using_train_data:
        train_dataset, _ = utils.datasets(arguments)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=len(train_dataset),
                                                  shuffle=False)
        encoder_matching_train(train_loader, arguments)
        ret = encoder_matching_test(test_loader, arguments)

    else:
        train_dataset, test_dataset = utils.datasets(arguments)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=arguments.train_loader_shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset),
                                                  shuffle=False)

        encoder_matching_train(train_loader, arguments)
        ret = encoder_matching_test(test_loader, arguments)

    return ret


def encoder_fitting(sae_A, sae_B, party_B_idc, train_loader, arguments):
    device = arguments.device
    # Training SAEs

    # Initialize model that fits E_A to E_B
    model = NNEncoder(arguments, arguments.encoding_size, arguments.encoding_size, arguments.hidden_F_size
                      , arguments.hidden_F_depth).to(device)

    check_safety_requirements(model, arguments)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate_pre)

    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs_pre):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)

            # Forward pass
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc]).detach()
            encoder_B = sae_B.encode(inputs[:, party_B_idc]).detach()
            outputs = model.forward(encoder_A)  # inputs[:, arguments.party_A_idc])
            loss_of_sae = criterion(outputs, encoder_B)

            # Backward and optimize: SAE
            optimizer.zero_grad()
            loss_of_sae.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('SAE-A2B-TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, arguments.num_epochs_pre, i + 1, total_step, loss_of_sae.item()))

    return model


def encoder_residual_fitting(sae_A2B, sae_A, sae_B, party_B_idc, train_loader, arguments):
    device = arguments.device
    # Training SAEs

    # Initialize model that fits A to B
    model = NNDisturber(arguments, len(party_B_idc), arguments.encoding_size).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate_pre)

    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs_pre):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)

            # Forward pass
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc]).detach()
            encoder_A2B = sae_A2B.encode(encoder_A).detach()
            encoder_B = sae_B.encode(inputs[:, party_B_idc]).detach()
            difference = encoder_B - encoder_A2B
            outputs = model.forward(inputs[:, party_B_idc])  # inputs[:, arguments.party_A_idc])
            loss_of_sae = criterion(arguments.eps * outputs, difference)

            # Backward and optimize: SAE
            optimizer.zero_grad()
            loss_of_sae.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('SAE-B2V-TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, arguments.num_epochs_pre, i + 1, total_step, loss_of_sae.item()))

    return model


def classifier_training(train_loader, arguments):
    device = arguments.device
    # Training classifier

    # Initialize model
    model = NNClassifier(arguments, arguments.encoding_size * (arguments.n_passives + 1)).to(device)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate,
                                 weight_decay=0)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass (with no param updates)
            encoder_A = arguments.sae_A.encode(inputs[:, arguments.party_A_idc])
            classifier_input = encoder_A
            for sae_A2B, sae_B2V, idc in zip(arguments.sae_A2Bs, arguments.sae_B2Vs, arguments.party_B_list):
                encoder_B_ = sae_A2B.encode(encoder_A)
                encoder_V = sae_B2V.generate(inputs[:, idc])
                # Forward pass
                classifier_input = torch.cat([classifier_input, encoder_B_ + arguments.eps * encoder_V], dim=1)

            c = model.forward(classifier_input)

            # Calculate Loss
            loss = ce_loss(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    return model


def encoder_matching_train(train_loader, arguments):
    if arguments.display_train_rec:
        print("\n\n----OUR MODEL----")

    # SAE pre-training
    sae_A = sae_trainer(arguments.party_A_idc, train_loader, arguments, 'A')
    sae_A2Bs, sae_B2Vs = [], []
    for party_B_idc in arguments.party_B_list:
        sae_B = sae_trainer(party_B_idc, train_loader, arguments, 'B')
        sae_A2B = encoder_fitting(sae_A, sae_B, party_B_idc, train_loader, arguments)
        sae_B2V = encoder_residual_fitting(sae_A2B, sae_A, sae_B, party_B_idc, train_loader, arguments)
        # sae_Bs.append(sae_B)
        sae_A2Bs.append(sae_A2B)
        sae_B2Vs.append(sae_B2V)

    arguments.sae_A = sae_A
    arguments.sae_A2Bs = sae_A2Bs
    arguments.sae_B2Vs = sae_B2Vs

    classifier_model = classifier_training(train_loader, arguments)

    # Save the model checkpoint
    arguments.classifier = classifier_model


def encoder_matching_test(test_loader, arguments, binary=True):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    sae_A = arguments.sae_A
    classifier_model = arguments.classifier

    bs = test_loader.batch_size
    label_pairs = [[], [], []]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            number = inputs[:, 0].long()
            inputs = inputs[:, 1:].to(device)
            # predicted = double_layer.test_forward(inputs, sae_A, sae_B, sae_A2B, sae_B2V, classifier_model, arguments)
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc])
            classifier_input = encoder_A
            for sae_A2B, sae_B2V, idc in zip(arguments.sae_A2Bs, arguments.sae_B2Vs, arguments.party_B_list):
                encoder_B_ = sae_A2B(encoder_A)
                encoder_B_ = torch.zeros_like(encoder_B_)
                encoder_V = sae_B2V.generate(inputs[:, idc])
                classifier_input = torch.cat([classifier_input, encoder_B_ + arguments.eps * encoder_V], dim=1)
            # encoder_V = sign_mapping((encoder_B - encoder_B_)/arguments.eps)
            predicted = classifier_model.predict(classifier_input)
            if arguments.record_classification:
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                label_pairs[0] += (number.tolist())
                label_pairs[1] += (predicted.tolist())
                label_pairs[2] += (labels.tolist())
            else:
                label_pairs[0] += (predicted.tolist())
                label_pairs[1] += (labels.tolist())
                label_pairs[2] += (labels.tolist())  # useless

    label_pairs = np.array(label_pairs).T
    if arguments.record_classification:
        results = utils.model_evaluation_rcr(label_pairs, "ours", arguments)
    else:
        if binary:
            results = utils.model_evaluation(label_pairs, "ours", arguments)
        else:
            results = utils.model_evaluation_multi(label_pairs, "ours", arguments)
    return results


def sign_mapping(x):
    return torch.sign(torch.round(x))


def input_diff(inputs1, inputs2):
    diff = torch.sub(inputs1, inputs2)
    diff = torch.pow(diff, 2)
    diff = torch.mean(diff, dim=1)

    return diff


def mul_input_diff(inputs1, inputs2):
    size_a = len(inputs1)
    size_b = len(inputs2)
    diff = torch.zeros(size_a, size_b)
    for i in range(size_a):
        for j in range(size_b):
            d = torch.sub(inputs1[i, :], inputs2[j, :])
            d = torch.pow(d, 2)
            d = torch.mean(d, dim=-1)
            diff[i, j] = d

    return diff
