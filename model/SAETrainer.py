import torch
from torch import nn
from model.Module import NNAutoEncoder


def sae_trainer(data_idc, train_loader, arguments, name=''):
    device = arguments.device
    # Training SAEs

    # Access data size
    data_size = len(data_idc)

    # Initialize Auto Encoder: NNAutoEncoder
    model = NNAutoEncoder(arguments, data_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate_pre)

    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs_pre):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:]
            inputs = inputs[:, data_idc].to(device)

            # Forward pass
            outputs = model.forward(inputs)
            loss = criterion(inputs, outputs)

            # Backward and optimize: SAE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('SAE-' + name + '-TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, arguments.num_epochs_pre, i + 1, total_step, loss.item()))

    return model


def sae_trainer_with_alignment(data_idc, t_data_idc, t_sae, train_loader, arguments, name=''):
    t_sae.eval()
    device = arguments.device
    lam = arguments.lam
    # Training SAEs

    # Access data size
    data_size = len(data_idc)

    # Initialize Auto Encoder: NNAutoEncoder
    model = NNAutoEncoder(arguments, data_size).to(device)

    # Loss and optimizer
    recon_criterion = nn.MSELoss()
    align_criterion = None
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate_pre)

    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs_pre):
        for i, (inputs, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            inputs = inputs[:, 1:]
            s_inputs = inputs[:, data_idc].to(device)
            t_inputs = inputs[:, t_data_idc].to(device)

            # Forward pass
            target = t_sae.encode(t_inputs)
            encoded = model.encode(s_inputs, req_grad=True)
            rec = model.decode(encoded)

            recon_loss = recon_criterion(s_inputs, rec)
            align_loss = torch.mean(target * encoded)
            loss = lam * recon_loss + (1 - lam) * align_loss

            # Backward and optimize: SAE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('SAE-' + name + '-TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, arguments.num_epochs_pre, i + 1, total_step, loss.item()))

    return model

