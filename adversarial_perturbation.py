import numpy as np
import deepfool
import torch
desired_accuracy = .3
import os

def project_perturbation(data_point,p,perturbation  ):

    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def generate(path,trainset, testset, net, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=20):
    '''
    :param path:
    :param dataset:
    :param testset:
    :param net:
    :param delta:
    :param max_iter_uni:
    :param p:
    :param num_class:
    :param overshoot:
    :param max_iter_df:
    :return:
    '''
    net.eval()
    device = 'cpu'



    v=np.zeros([224,224,3])
    fooling_rate = 0.0
    itr = 0

    # start an epoch
    while fooling_rate < 1-delta and itr < max_iter_uni:
        print("Starting pass number ", itr)
        k = 0
        dataiter = iter(trainset)
        for images in dataiter.next():
            r2 = int(net(images).max(1)[1])
            torch.cuda.empty_cache()


            r1 = int(net((images+v.astype(np.uint8))).max(1)[1])
            torch.cuda.empty_cache()

            if r1 == r2:
                dr, iter_k, label, k_i, pert_image = deepfool((images+v.astype(np.uint8)), net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                if iter_k < max_iter_df-1:

                    v[:, :, 0] += dr[0, 0, :, :]
                    v[:, :, 1] += dr[0, 1, :, :]
                    v[:, :, 2] += dr[0, 2, :, :]
                    v = project_perturbation(xi, p, v)

        itr = itr + 1

        with torch.no_grad():
            # Compute fooling_rate
            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))

            batch = 32


            test_loader_orig = testset
            i = 0
            for batch_idx, (inputs, _) in enumerate(testset):
                i +=1
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
            torch.cuda.empty_cache()

            for batch_idx, (inputs, _) in enumerate(testset):
                inputs = inputs.to(device)
                inputs += v
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))
            torch.cuda.empty_cache()

            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert))/float(i)
            print("FOOLING RATE: ", fooling_rate)
            np.save('v'+str(itr)+'_'+str(round(fooling_rate, 4)), v)

    return v
