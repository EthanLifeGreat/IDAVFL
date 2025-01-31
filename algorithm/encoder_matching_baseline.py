from algorithm.encoder_matching import *


def encoder_matching_baseline(arguments):
    if arguments.test_using_train_data:
        train_dataset, _ = utils.datasets(arguments.data)
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
        train_dataset, test_dataset = utils.datasets(arguments.data)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=arguments.train_loader_shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset),
                                                  shuffle=False)
        encoder_matching_train_baseline(train_loader, arguments)
        encoder_matching_test_baseline(test_loader, arguments)


def encoder_matching_train_baseline(train_loader, arguments):
    if arguments.display_train_rec:
        print("\n\n----BASELINE----")
    # Device configuration
    device = arguments.device

    # Initialize model
    model = NNClassifier(arguments, arguments.input_A_size).to(device)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass
            c = model.forward(inputs[:, arguments.party_A_idc])

            # Calculate Loss
            loss = ce_loss(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), arguments.model_path + 'baseline_model.ckpt')


def encoder_matching_train_baseline_sae(train_loader, arguments):
    if arguments.display_train_rec:
        print("\n\n----BASELINE SAE----")
    # Device configuration
    device = arguments.device

    # Initialize model
    classifier_model = NNClassifier(arguments, arguments.encoding_size + arguments.input_A_size).to(device)

    # SAE pre-training
    sae_A = sae_trainer(arguments.party_A_idc, train_loader, arguments)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=arguments.learning_rate,
                                 weight_decay=0)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass (with no param updates)
            sae_A_output = sae_A.encode(inputs[:, arguments.party_A_idc])

            # Forward pass
            classifier_input = torch.cat([inputs[:, arguments.party_A_idc], sae_A_output], dim=1)
            c = classifier_model.forward(classifier_input)

            # Calculate Loss
            loss = ce_loss(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(sae_A.state_dict(), arguments.model_path + 'baseline_sae_model.ckpt')
    torch.save(classifier_model.state_dict(), arguments.model_path + 'baseline_classifier_model.ckpt')


def encoder_matching_train_baseline_sae_high(train_loader, arguments):
    if arguments.display_train_rec:
        print("\n\n----BASELINE SAE HIGH----")
    # Device configuration
    device = arguments.device

    # Initialize model
    classifier_model = NNClassifier(arguments, arguments.encoding_size).to(device)

    # SAE pre-training
    sae_all = sae_trainer(arguments.all_idc, train_loader, arguments)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=arguments.learning_rate,
                                 weight_decay=0)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass (with no param updates)
            sae_output = sae_all.encode(inputs[:, arguments.all_idc])

            # Forward pass
            classifier_input = sae_output
            c = classifier_model.forward(classifier_input)

            # Calculate Loss
            loss = ce_loss(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(sae_all.state_dict(), arguments.model_path + 'baseline_high_sae_model.ckpt')
    torch.save(classifier_model.state_dict(), arguments.model_path + 'baseline_high_sae_classifier_model.ckpt')


def encoder_matching_train_baseline_high(train_loader, arguments):
    if arguments.display_train_rec:
        print("\n\n----BASELINE (HIGH)----")
    # Device configuration
    device = arguments.device

    # Initialize model
    model = NNClassifier(arguments, arguments.input_A_size + arguments.input_B_size).to(device)
    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:].to(device)
            labels = labels.to(device)

            # Forward pass
            c = model.forward(inputs)

            # Calculate Loss
            loss = ce_loss(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), arguments.model_path + 'baseline_high_model.ckpt')


def encoder_matching_test_baseline(test_loader, arguments):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    model = NNClassifier(arguments, arguments.input_A_size).to(device)
    model.load_state_dict(torch.load(arguments.model_path + 'baseline_model.ckpt'))

    bs = test_loader.batch_size
    if arguments.record_classification:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 3]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                number = inputs[:, 0].long()
                inputs = inputs[:, 1:].to(device)
                predicted = model.predict(inputs[:, arguments.party_A_idc])
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                label_pairs[:, 0] = number
                label_pairs[:, 1] = predicted
                label_pairs[:, 2] = labels

        results = utils.model_evaluation_rcr(label_pairs, "baseline", arguments)
    else:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 2]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs[:, 1:].to(device)
                predicted = model.predict(inputs[:, arguments.party_A_idc])
                label_pairs[:, 0] = predicted
                label_pairs[:, 1] = labels

        results = utils.model_evaluation(label_pairs, "baseline", arguments)

    return results


def encoder_matching_test_baseline_sae(test_loader, arguments):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    sae_A = NNAutoEncoder(arguments, arguments.input_A_size).to(device)
    classifier_model = NNClassifier(arguments, arguments.encoding_size + arguments.input_A_size).to(device)
    sae_A.load_state_dict(torch.load(arguments.model_path + 'baseline_sae_model.ckpt'))
    classifier_model.load_state_dict(torch.load(arguments.model_path + 'baseline_classifier_model.ckpt'))

    bs = test_loader.batch_size
    if arguments.record_classification:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 3]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                number = inputs[:, 0].long()
                inputs = inputs[:, 1:].to(device)
                sae_A_output = sae_A.encode(inputs[:, arguments.party_A_idc])
                classifier_input = torch.cat([inputs[:, arguments.party_A_idc], sae_A_output], dim=1)
                predicted = classifier_model.predict(classifier_input)
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                label_pairs[:, 0] = number
                label_pairs[:, 1] = predicted
                label_pairs[:, 2] = labels

        results = utils.model_evaluation_rcr(label_pairs, "baseline_sae", arguments)
    else:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 2]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs[:, 1:].to(device)
                sae_A_output = sae_A.encode(inputs[:, arguments.party_A_idc])
                classifier_input = torch.cat([inputs[:, arguments.party_A_idc], sae_A_output], dim=1)
                predicted = classifier_model.predict(classifier_input)
                label_pairs[:, 0] = predicted
                label_pairs[:, 1] = labels

        results = utils.model_evaluation(label_pairs, "baseline_sae", arguments)

    return results


def encoder_matching_test_baseline_sae_high(test_loader, arguments):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    sae_all = NNAutoEncoder(arguments, arguments.num_features).to(device)
    classifier_model = NNClassifier(arguments, arguments.encoding_size).to(device)
    sae_all.load_state_dict(torch.load(arguments.model_path + 'baseline_high_sae_model.ckpt'))
    classifier_model.load_state_dict(torch.load(arguments.model_path + 'baseline_high_sae_classifier_model.ckpt'))

    bs = test_loader.batch_size
    if arguments.record_classification:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 3]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                number = inputs[:, 0].long()
                inputs = inputs[:, 1:].to(device)
                sae_output = sae_all.encode(inputs[:, arguments.all_idc])
                classifier_input = sae_output
                predicted = classifier_model.predict(classifier_input)
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                label_pairs[:, 0] = number
                label_pairs[:, 1] = predicted
                label_pairs[:, 2] = labels

        results = utils.model_evaluation_rcr(label_pairs, "baseline_sae_high", arguments)
    else:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 2]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs[:, 1:].to(device)
                sae_output = sae_all.encode(inputs[:, arguments.all_idc])
                classifier_input = sae_output
                predicted = classifier_model.predict(classifier_input)
                label_pairs[:, 0] = predicted
                label_pairs[:, 1] = labels

        results = utils.model_evaluation(label_pairs, "baseline_sae_high", arguments)

    return results


def encoder_matching_test_baseline_high(test_loader, arguments):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    model = NNClassifier(arguments, arguments.input_A_size + arguments.input_B_size).to(device)
    model.load_state_dict(torch.load(arguments.model_path + 'baseline_high_model.ckpt'))

    bs = test_loader.batch_size
    if arguments.record_classification:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 3]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                number = inputs[:, 0].long()
                inputs = inputs[:, 1:].to(device)
                predicted = model.predict(inputs)
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                label_pairs[:, 0] = number
                label_pairs[:, 1] = predicted
                label_pairs[:, 2] = labels

        results = utils.model_evaluation_rcr(label_pairs, "baseline_high", arguments)
    else:
        with torch.no_grad():
            label_pairs = torch.zeros([bs, 2]).long()
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs[:, 1:].to(device)
                predicted = model.predict(inputs)
                label_pairs[:, 0] = predicted
                label_pairs[:, 1] = labels

        results = utils.model_evaluation(label_pairs, "baseline_high", arguments)

    return results
