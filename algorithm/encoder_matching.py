import sys
import warnings

import numpy as np
import utils.utils as utils
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
        encoder_matching_test(test_loader, arguments)

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
        encoder_matching_test(test_loader, arguments)


def encoder_fitting(sae_A, sae_B, train_loader, arguments):
    device = arguments.device
    # Training SAEs

    # Initialize model that fits A to B
    model = NNEncoder(arguments, arguments.encoding_size, arguments.encoding_size, arguments.hidden_F_size
                      , arguments.hidden_F_depth).to(device)

    # num of model params should be smaller than num of encoders to meet SAFETY requirements
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
            encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc]).detach()
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


def encoder_residual_fitting(sae_A2B, sae_A, sae_B, train_loader, arguments):
    device = arguments.device
    # Training SAEs

    # Initialize model that fits A to B
    model = NNDisturber(arguments, arguments.input_B_size, arguments.encoding_size).to(device)

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
            encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc]).detach()
            difference = encoder_B - encoder_A2B
            outputs = model.forward(inputs[:, arguments.party_B_idc])  # inputs[:, arguments.party_A_idc])
            loss_of_sae = criterion(arguments.eps * outputs, difference)

            # Backward and optimize: SAE
            optimizer.zero_grad()
            loss_of_sae.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('SAE-B2V-TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, arguments.num_epochs_pre, i + 1, total_step, loss_of_sae.item()))

    return model


def classifier_training(sae_A2B, sae_A, sae_B2V, train_loader, arguments):
    device = arguments.device
    # Training classifier

    # Initialize model
    model = NNClassifier(arguments, arguments.encoding_size * 2).to(device)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate,
                                 weight_decay=0)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):  # * 10
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass (with no param updates)
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc])
            # encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc])
            encoder_B_ = sae_A2B.encode(encoder_A)
            encoder_V = sae_B2V.generate(inputs[:, arguments.party_B_idc])
            # encoder_V = sign_mapping((encoder_B - encoder_B_)/arguments.eps)

            # Forward pass
            classifier_input = torch.cat([encoder_A, encoder_B_ + arguments.eps * encoder_V], dim=1)
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
    sae_B = sae_trainer(arguments.party_B_idc, train_loader, arguments, 'B')
    # wae_A, wae_B = double_layer.ws_double_sae_trainer(sae_A, sae_B, train_loader, arguments)
    sae_A2B = encoder_fitting(sae_A, sae_B, train_loader, arguments)
    sae_B2V = encoder_residual_fitting(sae_A2B, sae_A, sae_B, train_loader, arguments)
    classifier_model = classifier_training(sae_A2B, sae_A, sae_B2V, train_loader, arguments)

    # Save the model checkpoint
    torch.save(sae_A.state_dict(), arguments.model_path + 'sae_A_model.ckpt')
    torch.save(sae_A2B.state_dict(), arguments.model_path + 'sae_A2B_model.ckpt')
    torch.save(classifier_model.state_dict(), arguments.model_path + 'classifier_model.ckpt')
    arguments.sae_A = sae_A
    arguments.sae_B = sae_B
    arguments.sae_A2B = sae_A2B
    arguments.sae_B2V = sae_B2V
    arguments.classifier = classifier_model


def encoder_matching_test(test_loader, arguments, binary=True):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    sae_A = arguments.sae_A
    sae_B = arguments.sae_B
    sae_A2B = arguments.sae_A2B
    sae_B2V = arguments.sae_B2V
    classifier_model = arguments.classifier

    bs = test_loader.batch_size
    label_pairs = [[], [], []]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            number = inputs[:, 0].long()
            inputs = inputs[:, 1:].to(device)
            # predicted = double_layer.test_forward(inputs, sae_A, sae_B, sae_A2B, sae_B2V, classifier_model, arguments)
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc])
            # encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc])
            encoder_B_ = sae_A2B(encoder_A)
            # encoder_B_ = torch.zeros_like(encoder_B_)
            encoder_V = sae_B2V.generate(inputs[:, arguments.party_B_idc])
            # encoder_V = sign_mapping((encoder_B - encoder_B_)/arguments.eps)
            classifier_input = torch.cat([encoder_A, encoder_B_ + arguments.eps * encoder_V], dim=1)
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


def encoder_matching_test_continuous(test_loader, arguments, binary=True):
    device = arguments.device

    sae_A = arguments.sae_A
    sae_B = arguments.sae_B
    sae_A2B = arguments.sae_A2B
    sae_B2V = arguments.sae_B2V
    classifier_model = arguments.classifier

    bs = test_loader.batch_size
    label_pairs = [[], [], []]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            number = inputs[:, 0].long()
            inputs = inputs[:, 1:].to(device)
            # predicted = double_layer.test_forward(inputs, sae_A, sae_B, sae_A2B, sae_B2V, classifier_model, arguments)
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc])
            # encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc])
            encoder_B_ = sae_A2B(encoder_A)
            # encoder_B_ = torch.zeros_like(encoder_B_)
            encoder_V = sae_B2V.generate(inputs[:, arguments.party_B_idc])
            # encoder_V = sign_mapping((encoder_B - encoder_B_)/arguments.eps)
            classifier_input = torch.cat([encoder_A, encoder_B_ + arguments.eps * encoder_V], dim=1)
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
    if binary:
        results = utils.model_evaluation_continuous(label_pairs, "ours", arguments)
    else:
        results = utils.model_evaluation_continuous_multi(label_pairs, "ours", arguments)
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


def check_safety_requirements(model, arguments):
    """
    num of model params should be smaller than num of encoders to meet SAFETY requirements
    check if such requirement is satisfied
    """
    num_model_params = sum(x.numel() for x in model.parameters())
    num_encoder_params = arguments.encoding_size * arguments.batch_size
    if num_model_params < num_encoder_params:
        print("Fitting Encoder has {} parameters in total, which is smaller than {}, the size of auto-encoders."
              "Thus safety requirements satisfied.".format(num_model_params, num_encoder_params), file=sys.__stdout__)
    else:
        warnings.warn("Fitting Encoder has {} parameters in total, which is greater or equal than {}.\n "
                      "Thus safety requirements NOT satisfied\n".format(num_model_params, num_encoder_params) +
                      "Please consider setting smaller the number of model F's params or greater the "
                      "dim of auto-encoders.")
