from torch.utils.data import DataLoader
from dataloader import *
from metric import *

def compute_metric(labels,func, f,*a):
    loss = 0
    device = labels.device
    positive = torch.zeros_like(f).to('cuda')
    for i in range(0,len(a)):
        positive_B = torch.zeros_like(f)
        positive_B [labels == i]=1
        positive += torch.mul(positive_B,a[i])

    weight_mask = torch.sparse_coo_tensor(size=(f.size()[0],len(a)), device=device).to_dense().to('cuda')
    anchor = f
    unique_label = torch.unique(labels)            
    for cls in unique_label:
        cls = cls.item()
        cls_inds = torch.nonzero(labels == cls).squeeze(1) 
        cur_labels = [cls]
        cur_labels = torch.tensor(cur_labels, device=device)
        tmp_weight_mask_vec = weight_mask[cls_inds]
        tmp_weight_mask_vec[:, cur_labels] = 1
        weight_mask[cls_inds] = tmp_weight_mask_vec

    loss = func(anchor, positive, 1-weight_mask, *a)
    return loss



def label_acc(out, label):
    label = label.to('cuda')
    _, predicts = torch.max(out.data, 1)
    correct = (predicts == label).sum().item()
    return correct

def eval(feature_extractor, generator, Cls, Discri, epoch, acc_best):
    ismydataset = test_data_V2F()
    valdataloader = DataLoader(ismydataset, batch_size=50, shuffle=False, num_workers=8)
    feature_extractor.eval()
    Cls.eval()
    total_test = 0.0
    count_test = 0.0
    for index, data in enumerate(valdataloader):
        a, f1, f2, label = data
        a, f1, f2 = a.to('cuda'), f1.to('cuda'), f2.to('cuda')
        label = label.to('cuda')
        total_test += a.size(0)
        a, f1, f2 = feature_extractor.test_forward(a, f1, f2)
        a, f1, f2 = generator.test_forward(a, f1, f2)
        predict = Cls.test_forward(a, f1, f2)
        count_test += label_acc(predict, 1-label)
    acc2 = count_test / total_test
    if acc2 > acc_best:
        acc_best = acc2
        states1 = {
            'feature': feature_extractor.state_dict(),
            'G': generator.state_dict(),
            'D': Discri.state_dict(),
            'C': Cls.state_dict(),
        }
        name = 'V2F' + str(epoch) + '_' + str(acc2) + '.pkl'
        torch.save(states1, name)
    print('V2F test acc : ', acc2, 'best acc', acc_best)
    return acc_best
