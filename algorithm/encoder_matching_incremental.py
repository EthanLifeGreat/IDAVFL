from model.Module import *
import numpy as np
import utils.utils as utils
import torch.nn.functional


class BiasLayer(nn.Module):
    def __init__(self, num_new_classes_in_total):
        self.size = num_new_classes_in_total
        super(BiasLayer, self).__init__()
        a = torch.ones([1, num_new_classes_in_total], requires_grad=True)
        # b = torch.zeros([1, num_new_classes_in_total], requires_grad=True)
        b = torch.zeros([1, 1], requires_grad=True)
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

    def forward(self, x):
        z1 = x[:, :-self.size]
        z2 = self.a * x[:, -self.size:] + self.b
        z = torch.cat([z1, z2], dim=1)
        return z

    def predict(self, x):
        with torch.no_grad():
            c = self.forward(x)
            _, predicted = torch.max(c.data, 1)
        return predicted


def classifier_fine_tune(classifier, train_loader, arguments, scale=1e-1):
    device = arguments.device
    sae_A2B, sae_A, sae_B2V = arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V
    sae_A2B.eval()
    sae_A.eval()
    sae_B2V.eval()

    # Initialize model
    model = classifier.train()

    # Loss and optimizer
    ce_loss_func = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate * scale,
                                 weight_decay=0)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs):
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
            loss = ce_loss_func(c, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    return model


def classifier_distillation_training(teacher_classifier, num_dis_classes, train_loader,
                                     arguments, lam):
    device = arguments.device
    sae_A2B, sae_A, sae_B2V = arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V
    sae_A2B.eval()
    sae_A.eval()
    sae_B2V.eval()

    temp = 2

    # Initialize model
    model = NNClassifier(arguments, arguments.encoding_size * 2).to(device)

    # Loss and optimizer
    ce_loss_func = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    dis_loss_func = nn.KLDivLoss(reduction="batchmean")
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
            with torch.no_grad():
                t = teacher_classifier.forward(classifier_input)
                t = torch.nn.functional.softmax(t / temp, dim=1)
            c = model.forward(classifier_input)
            c1 = torch.nn.functional.log_softmax(c[:, :num_dis_classes] / temp, dim=1)

            # Calculate Loss
            dis_loss = dis_loss_func(c1, t)
            cla_loss = ce_loss_func(c, labels)
            loss = lam * dis_loss + (1 - lam) * cla_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    return model


def classifier_distillation_training_CI(teacher_classifier, num_dis_classes, train_loader,
                                     arguments, lam=None):
    device = arguments.device
    sae_A2B, sae_A, sae_B2V = arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V
    sae_A2B.eval()
    sae_A.eval()
    sae_B2V.eval()

    # lam = 0.0001

    if lam is None:
        lam = num_dis_classes / arguments.num_classes
    temp = 2

    # Initialize model
    model = NNClassifier(arguments, arguments.encoding_size * 2).to(device)

    # Loss and optimizer
    ce_loss_func = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    dis_loss_func = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate,
                                 weight_decay=0)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(arguments.num_epochs): #  * 10
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
            with torch.no_grad():
                t = teacher_classifier.forward(classifier_input)
                t = torch.nn.functional.softmax(t / temp, dim=1)
            c = model.forward(classifier_input)
            c_oc_logit = torch.nn.functional.log_softmax(c / temp, dim=1)[:, :num_dis_classes]

            # Calculate Loss
            dis_loss = dis_loss_func(c_oc_logit, t)
            cla_loss = ce_loss_func(c, labels)
            loss = lam * dis_loss + (1 - lam) * cla_loss

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    return model


def bic_training(classifier, num_inc_labels, train_loader, arguments):
    device = arguments.device
    sae_A2B, sae_A, sae_B2V = arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V
    sae_A2B.eval()
    sae_A.eval()
    sae_B2V.eval()
    classifier.eval()

    # Initialize model
    model = BiasLayer(num_inc_labels).to(device)

    # Loss and optimizer
    ce_loss_func = nn.CrossEntropyLoss(weight=arguments.weight_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(1):
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
            classifier_input = torch.cat([encoder_A, encoder_B_ + arguments.eps * encoder_V], dim=1)
            c = classifier.forward(classifier_input)

            # Forward pass
            out = model.forward(c)

            # Calculate Loss
            loss = ce_loss_func(out, labels)

            # Backward and optimize: Classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step) == 0 and arguments.display_train_rec:
                print('TRAINING: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, arguments.num_epochs, i + 1, total_step, loss.item()))

    return model


def encoder_matching_test_with_bic(test_loader, arguments, binary=False):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    # Device configuration
    device = arguments.device

    sae_A = arguments.sae_A
    sae_B = arguments.sae_B
    sae_A2B = arguments.sae_A2B
    sae_B2V = arguments.sae_B2V
    classifier_model = arguments.classifier
    bic = arguments.bic

    label_pairs = [[], [], []]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            number = inputs[:, 0].long()
            inputs = inputs[:, 1:].to(device)
            encoder_A = sae_A.encode(inputs[:, arguments.party_A_idc])
            # encoder_B = sae_B.encode(inputs[:, arguments.party_B_idc])
            encoder_B_ = sae_A2B(encoder_A)
            encoder_V = sae_B2V.generate(inputs[:, arguments.party_B_idc])
            # encoder_V = sign_mapping((encoder_B - encoder_B_)/arguments.eps)
            classifier_input = torch.cat([encoder_A, encoder_B_ + arguments.eps * encoder_V], dim=1)
            bic_input = classifier_model.forward(classifier_input)
            predicted = bic.predict(bic_input)
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
