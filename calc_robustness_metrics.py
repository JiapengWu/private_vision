import torch
import argparse
import os
import timm
import time
from tqdm import tqdm
import torchvision
from private_vision import Smooth
import torchvision.transforms as transforms
import foolbox as fb

def create_attack(model, name, kwargs=None):
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    if name.lower() == "fgsm": 
        return fmodel, fb.attacks.FGSM(), "linf"
    elif name.lower() == "pgd":
        return fmodel, fb.attacks.PGD(rel_stepsize=kwargs.get("PGD_stepsize", 2/255) , 
                                        steps=kwargs.get("PGD_iterations", 40)),  "linf"
    elif name.lower() == "pgdl2":
        return fmodel, fb.attacks.L2PGD(rel_stepsize=kwargs.get("PGDL2_stepsize", 0.2),
                                        steps=kwargs.get("PGDL2_iterations", 40)) , "l2"
    elif name.lower() == "cw":
        return fmodel, fb.attacks.L2CarliniWagnerAttack(binary_search_steps=kwargs.get("CW_c_iterations", 9), 
                                                        steps=kwargs.get("CW_iterations", 10000), 
                                                        stepsize=kwargs.get("CW_stepsize", 0.01), 
                                                        confidence=kwargs.get("CW_confidence", 0), 
                                                        initial_const=kwargs.get("CW_c", 1)), "l2" # abort_early=True
    elif name.lower() == "deepfooll2":
        return fmodel, fb.attacks.L2DeepFoolAttack(steps=kwargs.get("DeepFool_iterations", 50),
                                                    overshoot=kwargs.get("DeepFool_overshoot", 0.02),
                                                    loss=kwargs.get("DeepFool_loss", 'logits')), "l2" # loss='crossentropy'
    elif name.lower() == "deepfoollinf":
        return fmodel, fb.attacks.LinfDeepFoolAttack(steps=kwargs.get("DeepFool_iterations", 50),
                                                    overshoot=kwargs.get("DeepFool_overshoot", 0.02),
                                                    loss=kwargs.get("DeepFool_loss", 'logits')),  "linf" # loss='crossentropy'
    elif name.lower() == "boundary":
        return fmodel, fb.attacks.BoundaryAttack(init_attack=fb.attacks.L2FMNAttack(), # SaltAndPepperNoiseAttack()
                                                steps=kwargs.get("BA_iterations", 25000)), "l2"
    
    # HopSkipJumpAttack, PointwiseAttack
    

def robustness_succ(model, dataloader, gammas_linf=[0.15], gammas_l2=[0.5], adversarial_attacks=[], logdir=None,
                    adv_attack_params={}, view_sample_images=False, device=None):
    # This is very important for the compatibility of torchattack and opacus
    # model.disable_hooks()
    # Here we cannot use torch.no_grad() because adv_data = atk(data, labels) requires gradient updates to input
    adv_robustness_on_corr = {}
    for name in adversarial_attacks:
        print(f"evaluating adv attack {name}")
        fmodel, atk, norm = create_attack(model, name, adv_attack_params)
        print(f"norm: {norm}, atk: {atk}, fmodel: {fmodel}")

        if norm == "l2":
            gammas = gammas_l2
        else:
            gammas = gammas_linf  # if norm isnt defined or linf

        correct_on_corr = torch.zeros_like(torch.tensor(gammas))
        total_on_corr = 0
        # norm of perturbation for p=2,infty
        l2_norm_all = [[] for _ in gammas]
        linfty_norm_all = [[] for _ in gammas]
        # where f(x + adv) != f(x) = y
        l2_norm_miss = [[] for _ in gammas]
        linfty_norm_miss = [[] for _ in gammas]

        for _batch_idx, (data, labels) in enumerate(tqdm(dataloader)):
            data, labels = data.to(device), labels.to(device)
            _, predicted_benign = torch.max(model(data), 1)

            # on subset of batch that model correctly predicts (bening)
            data_corr = data[predicted_benign == labels]
            labels_corr = labels[predicted_benign == labels]

            if data_corr.shape[0] == 0:
                continue

            # create attack images
            _, adv_data, adv_success = atk(fmodel, data_corr, labels_corr, epsilons=gammas)
            labels_corr, adv_data, adv_success = labels_corr.to(device), [gamma.to(device) for gamma in adv_data], [
                gamma.to(device) for gamma in adv_success]
            # success of attack = 1 - accuracy of model on adv examples
            total_on_corr += labels_corr.size(0)
            correct_on_corr += torch.tensor([gamma.sum().item() for gamma in adv_success])

            for i in range(len(gammas)):
                l2_norm_all[i].extend([torch.linalg.vector_norm(x, ord=2).item() for x in (data_corr - adv_data[i])])
                linfty_norm_all[i].extend(
                    [torch.linalg.vector_norm(x, ord=float('inf')).item() for x in (data_corr - adv_data[i])])
                l2_norm_miss[i].extend(
                    [torch.linalg.vector_norm(x, ord=2).item() for x in (data_corr - adv_data[i])[adv_success[i]]])
                linfty_norm_miss[i].extend([torch.linalg.vector_norm(x, ord=float('inf')).item() for x in
                                            (data_corr - adv_data[i])[adv_success[i]]])

            if view_sample_images and _batch_idx == 0:
                # TODO: check data_corr isnt empty
                # TODO: visualize only successful adv examples
                for i, gamma in enumerate(gammas):
                    n = 10
                    outputs = model(adv_data[i][0:n])
                    _, predicted_adv = torch.max(outputs, 1)
                    view_adv_images(data_corr, adv_data[i][0:n], labels_corr, predicted_adv, f"{name}_{gamma}", logdir, n=n)

        adv_robustness_on_corr[name] = [(correct_on_corr[i] / total_on_corr).item() for i in range(len(gammas))]
        # l2 and linfty norm of perturbations (where f(x) = y)
        avg_perturb_2norm = [round(torch.mean(torch.tensor(l2_norm_all[i])).item(), 5) for i in range(len(gammas))]
        avg_perturb_inftynorm = [round(torch.mean(torch.tensor(linfty_norm_all[i])).item(), 5) for i in
                                 range(len(gammas))]
        # l2 and linfty norm of perturbations (where f(x+fdelta) =/= f(x) = y)
        avg_perturb_2norm_miss = [round(torch.mean(torch.tensor(l2_norm_miss[i])).item(), 5) for i in
                                  range(len(gammas))]
        avg_perturb_inftynorm_miss = [round(torch.mean(torch.tensor(linfty_norm_miss[i])).item(), 5) for i in
                                      range(len(gammas))]

        print(f"adv robust succ on {name}: {adv_robustness_on_corr[name]}, l2 norm: {avg_perturb_2norm}, linfty norm: {avg_perturb_inftynorm}, l2 norm miss: {avg_perturb_2norm_miss}, linfty norm miss: {avg_perturb_inftynorm_miss}")

    # Don't forget to turn them back on
    # model.enable_hooks()
    # the tensorflow writer requires scalar
    return [adv_robustness_on_corr[key][-1] for key in
            adversarial_attacks]  # NOTE: note this is only for last gamma, do not use


def acc(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        print(len(testloader), f'Acc: {acc}')
        
def certified_robustness(model, dataloader, num_classes, batch_size, **kwargs):
    smoothed_classifier = Smooth(model, num_classes, kwargs['certified_noise_std'], device)
    results = []
    before_time = time.time()
    for _, (data_point, label) in enumerate(tqdm(dataloader)):
        data_point, label = data_point.to(device), label.to(device)
        prediction, radius = smoothed_classifier.certify(data_point, kwargs['certified_n0'], kwargs['certified_n'],
                                                         kwargs['certified_alpha'], batch_size)
        correct = int(prediction == label)
        results.append([label.item(), prediction, radius, correct])
        # print(time.time() - before_time)
    after_time = time.time()
    time_elapsed = str(after_time - before_time)
    print(f"Seconds required to certify %d datapoints: " % len(dataloader) + time_elapsed)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluating certified robustness')
    parser.add_argument('--metric', default="acc", type=str)
    parser.add_argument('--checkpoint_path', type=str, default='CIFAR10')
    parser.add_argument('--batch_size', type=int, default=1800)
    args = parser.parse_args()

    print(args.checkpoint_path)
    checkpoint_split = args.checkpoint_path.split("_CIFAR")
    model = checkpoint_split[0]
    dataset = 'CIFAR' + checkpoint_split[1].split("_")[0]

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if dataset=='CIFAR10':
      testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
      num_classes=10 
    elif dataset=='CIFAR100':
      testset = torchvision.datasets.CIFAR100(
        root='data', train=False, download=True, transform=transform_test)
      num_classes=100
      
    device=torch.device("cuda")

    # adversarial_attacks = ["pgd", "fgsm", "pgdl2", "deepfooll2", "deepfoollinf", "cw", "boundary"]
    adversarial_attacks = ["fgsm"]
    gammas_linf = [0.0005, 0.0008, 0.0015, 0.003, 0.01, 0.05, 0.1, 0.15, 0.3, 0.4, 0.5, 0.6]
    gammas_l2 = [0.3, 0.5, 1, 2, 3, 4]
    attack_params ={
        "PGD_iterations": 40,
        "PGD_stepsize": 2 / 255,  # alpha in PGD
        "PGDL2_iterations": 40,
        "PGDL2_stepsize": 0.2,  # alpha in PGDL2
        "CW_c_iterations": 9,
        "CW_c": 1,
        "CW_confidence": 0,  # confidence
        "CW_iterations": 10000,
        "CW_stepsize": 0.01,
        "DeepFool_iterations": 50,
        "DeepFool_overshoot": 0.02,
        "DeepFool_loss": "logits",  # "logits", "crossentropy"
    }

    # testset.data = torch.tensor(testset.data).to(device)
    # testset.targets = torch.tensor(testset.targets).to(device)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = timm.create_model(model, pretrained=False, num_classes=num_classes)
    net = net.to(device)
    # load model state dict into net
    checkpoint = torch.load(args.checkpoint_path)
    net.load_state_dict(checkpoint)
    
    net.eval()
    if args.metric == "acc":
        acc(net, testloader)
    elif args.metric == "attack":
        robustness_succ(net, testloader, gammas_linf=gammas_linf, gammas_l2=gammas_l2, adversarial_attacks=adversarial_attacks, adv_attack_params=attack_params, device=device)
    else:
        with torch.no_grad():    
            certified_robustness(net, testloader, num_classes, args.batch_size, certified_n0=100, certified_n=10000, certified_alpha=0.001, certified_noise_std=0.1, device=device)
