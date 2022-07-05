import  torch, os
import  numpy as np
import  argparse

from meta_PN import Meta
from dataloader_train import CIFARGenerator as Loader_train
from dataloader_test import CIFARGenerator as Loader_test
from functions import *
from tqdm import tqdm
from tensorboardX import SummaryWriter

torch.set_num_threads(1)

train_path = '/drive1/YH/datasets/cifar-100-python/train'
test_path = '/drive1/YH/datasets/cifar-100-python/test'
writer = SummaryWriter()

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 3, 3, 3, 1, 0]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('bn', [64]),
        ('relu', [True]),
        # ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('bn', [64]),
        ('relu', [True]),
        # ('max_pool2d', [2, 2, 0]),
        ('avg_pool2d', [2, 1, 0])
        # ('flatten', []),
        # ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda:'+str(args.gpu))
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    loader_train = Loader_train(train_path=train_path, test_path = test_path, args = args, max_iter=10001)
    loader_test = Loader_test(train_path=train_path, test_path = test_path, args = args, max_iter=30)

    print("Data Loading Completed!")

    for step, x_spt, y_spt, x_qry, y_qry in tqdm(loader_train):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs = maml(step, x_spt, y_spt, x_qry, y_qry, device)

        if step % 40 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step >= 9000 and step % 100 == 0:
            if not os.path.isdir('save'):
                os.makedirs('save')
            torch.save(maml.state_dict(), 'save/%d_pth' % (step))

        if step % 400 == 0:  # evaluation
            accs_all_test = []
            loader_test.num_iter = 0
            for _, x_spt, y_spt, x_qry, y_qry  in loader_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, device)
                accs_all_test.append(accs)

            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('step:', step, 'Test acc:', accs)
            writer.add_scalar('Test Acc', accs, step)

    accs_all_test = []
    loader_test = Loader_test(train_path=train_path, test_path=test_path, args=args, max_iter=5000)

    for val_step, x_spt, y_spt, x_qry, y_qry in loader_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, device)
        accs_all_test.append(accs)
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float)
    print('step:', 5000, 'Test acc:', accs)


if __name__ == '__main__':


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--n_spt', type=int, help='n shot for support set', default=6)
    argparser.add_argument('--n_qry', type=int, help='n shot for query set', default=6)
    argparser.add_argument('--local_ep', type=int, help='n shot for query set', default=1)
    argparser.add_argument('--total_user',type=int, default=64)
    argparser.add_argument('--test_total_user', type=float, default=10)
    argparser.add_argument('--n_user',type=float, default=10)
    argparser.add_argument('--n_split',type=int, default=2)
    argparser.add_argument('--test_n_split',type=int, default=4)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=64)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=10)
    argparser.add_argument('--gpu', type=int, help='gpu number', default=4)
    argparser.add_argument('--round', type=int, help='round number', default=3)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=60)
    argparser.add_argument('--test_batch_size', type=int, help='n_class in episode in FT', default=30)
    argparser.add_argument('--n_cls_at_test', type=int, help='n_class in episode in FT', default=5)
    argparser.add_argument('--data_ratio', type=float,  default=1)

    args = argparser.parse_args()

    main()
