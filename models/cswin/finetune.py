from dataset import ABAP
from pick_models import CSWin_tiny
from evaluation import Eval

import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
import shutil
import warnings
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(
        prog='CSWin fine-tune',
        description='CSWin fine-tune',
    )

parser.add_argument('--device', default='cuda', help='device', choices=['cuda', 'cpu'])
parser.add_argument('--seed', default=2023, help='seed', type=int)
parser.add_argument('--result_dir', default='./results/', help='results path')

# data
parser.add_argument("--dataset", default='abap', help='dataset', choices=['abap', 'gsv', 'gsv_unlabel'])
parser.add_argument("--aug", help="balanced data", action="store_true")
parser.add_argument('--trainsize', default=256, type=int, help='train batch size')
parser.add_argument('--testsize', default=128, type=int, help='test batch size')

# model
parser.add_argument('--numofcat', default=4, type=int, help='classifier')
parser.add_argument("--weighted", help="weighted loss", action="store_true")
# parser.add_argument('--pretrain', default=True, type=bool, help='pretrain model')
# parser.add_argument('--fixed', default=True, type=bool, help='fix model parameters')

# optimization
parser.add_argument('--opt', default='adam', help='Optimizer', choices=['sgd', 'adam'])
parser.add_argument('--lr', default=0.001,type=float, help='Learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='Momentum')

parser.add_argument('--epochs', default=2, type=int, help='epoch')

# phase
parser.add_argument('--phase', default='classifier', help='training phase', choices=['classifier', 'fine-tune', 'eval', 'grad_cam', 'test'])
parser.add_argument('--path', default='yourpath.pth.tar', help='model path')
parser.add_argument("--vis", help="Visualization", action="store_true")

DATA_PATH = os.environ.get('MVIT_PATH', 'yourpath')

def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    warnings.warn('You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )
    
def get_result_dir(args):
    result_dir = args.result_dir
    if args.aug:
        result_dir = result_dir + 'augment_'
        if args.weighted:
            raise NotImplementedError
    else:
        if args.weighted:
            result_dir = result_dir + 'weighted_'

    if args.phase == 'classifier':
        result_dir = result_dir + 'cls_'
    
    if args.phase == 'fine-tune':
        result_dir = result_dir + 'ft_'
    
    if args.opt == 'adam':
        result_dir = result_dir + f'{args.opt}_lr{args.lr}'
    elif args.opt == 'sgd':
        result_dir = result_dir + f'{args.opt}_lr{args.lr}_mom{args.mom}'
    return result_dir

def create_result_dir(result_dir):
    id = 0
    while True:
        result_dir_id = result_dir + '_id%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    os.makedirs(result_dir_id + '/checkpoints')
    return result_dir_id

def save_checkpoint(state, is_best, result_dir, epoch, filename='checkpoint.pth.tar'):
    filename = result_dir + f'/checkpoints/epoch@{epoch}_' + filename

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, result_dir + '/' + f'model_best@{epoch}.pth.tar')

def make_vis_folder(result_dir, atepoch, dataset):
    vis_folder = result_dir + f'VIS{atepoch}_{dataset}'
    os.makedirs(vis_folder)
    for i in range(1, 5):
        os.makedirs(vis_folder + f'/class_{i}')
        for j in range(1, 5):
            os.makedirs(vis_folder + f'/class_{i}' + f'/class_{j}')
    return vis_folder

def move_image(pred, label, vis_folder, sample_filename):
    shutil.copyfile(sample_filename, vis_folder + f'/class_{label+1}' + f'/class_{pred+1}' + f'/{sample_filename.split("/")[-1]}')
    return


def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for _, (img, y) in enumerate(train_loader):
        inputs = img.to(device)
        targets = y.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, targets)
        
        outputs['loss'].backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += outputs['loss'].data.item() * targets.size(0)
            total += targets.size(0)
            correct += sum((outputs['predictions'].argmax(1) == targets).float()).detach().cpu().numpy()
        break
    return train_loss / total, 100. * correct / total


def test(model, test_loader,  device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (img, y) in enumerate(test_loader):
            inputs = img.to(device)
            targets = y.to(device)


            outputs = model(inputs, targets)

            test_loss += outputs['loss'].data.item() * targets.size(0)
            total += targets.size(0)
            correct += sum((outputs['predictions'].argmax(1) == targets).float()).detach().cpu().numpy()
            break
    return test_loss / total, 100. * correct / total

def eval(model, test_loader, device, vis, vis_folder, numofcat):
    model.eval()
    collect_labels = []
    collect_decisions = []
    with torch.no_grad():
        for i, (img, y) in enumerate(tqdm(test_loader)):
            inputs = img.to(device)
            targets = y.to(device)


            outputs = model(inputs, targets)
            # print(int(outputs['predictions'].argmax(1)), int(y))
            collect_labels.append(targets.cpu().numpy())
            collect_decisions.append(outputs['predictions'].argmax(1).cpu().numpy())
            if vis:
                sample_fname, _ = test_loader.dataset.samples[i]
                move_image(int(outputs['predictions'].argmax(1)), int(y), vis_folder, sample_fname)
    all_decisions =  np.concatenate(collect_decisions)
    all_labels = np.concatenate(collect_labels)
    # print(all_decisions[:10], all_labels[:10])
    metrics = Eval(all_decisions, all_labels, numofcat=numofcat)
    return metrics

def test(model, test_loader, device, test_folder, numofcat):
    model.eval()
    with torch.no_grad():
        for i, (img, y) in enumerate(tqdm(test_loader)):
            inputs = img.to(device)
            targets = y.to(device)


            outputs = model(inputs, targets)
            
            sample_fname, _ = test_loader.dataset.samples[i]
            res_dict = {'predictions': int(outputs['predictions'].argmax(1)), 'fname': sample_fname.split('/')[-1]}
            with open(test_folder + sample_fname.split('/')[-1].split('.')[0] + '.json', "w") as outfile:
                json.dump(res_dict, outfile)

def main():
    args = parser.parse_args()
    random_seed(args.seed)

    if args.phase == 'classifier' or args.phase == 'fine-tune':

        result_dir = get_result_dir(args)
        result_dir = create_result_dir(result_dir)

        dataset = ABAP(args.dataset, DATA_PATH, args.trainsize, args.testsize, args.aug)
        dataset.prepare_data()
        train_loader, test_loader = dataset.dataloaders()

        if args.path is None:
            raise NotImplementedError

        if args.phase == 'classifier':
            model = CSWin_tiny(args.numofcat, weighted=args.weighted, path=args.path, fixed=True, device=args.device)
        else:
            model = CSWin_tiny(args.numofcat, weighted=args.weighted, path=args.path, fixed=False, device=args.device)
        
        model.to(args.device)

        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        elif args.opt == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        else:
            raise NotImplementedError
        
        dfhistory = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'], dtype=np.float16)
        best_acc = 0
        for epoch in range(1, 1 + args.epochs):
            print('Training...')
            train_loss, train_acc = train(model, train_loader, optimizer, args.device)
            print('Test...')
            test_loss, test_acc = test(model, test_loader, args.device)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            info = (int(epoch), train_loss, train_acc, test_loss, test_acc)
            dfhistory.loc[epoch-1] = info
            dfhistory.to_csv(f'{result_dir}/history.csv', index=False)


            state = {
                'epoch': epoch, 
                'arch': 'CSWin_tiny', 
                'state_dict': model.state_dict(),
                'best_acc': best_acc, 
                'optimizer' : optimizer.state_dict(),
            }
        
            save_checkpoint(state, is_best, result_dir, epoch)

            print(f'Epoch = {epoch}')
            print(f'train_loss = {train_loss}, train_acc = {train_acc}')
            print(f'test_loss = {test_loss}, test_acc = {test_acc}')
    
    elif args.phase == 'eval':
        result_dir = args.path.split('model_best')[0]
        atepoch = args.path.split('model_best')[-1].replace('.pth.tar', '')
        if glob(f'{result_dir}metrics{atepoch}_{args.dataset}.json') and not args.vis:
            f = open(f'{result_dir}metrics{atepoch}_{args.dataset}.json') 
            print(json.load(f))

        else:
            print('vis-go')
            if args.vis:
                vis_folder = make_vis_folder(result_dir, atepoch, args.dataset)
            else:
                vis_folder = ''
            dataset = ABAP(args.dataset, DATA_PATH, trainsize=1, testsize=1)
            dataset.prepare_data()
            # test_loader, _ = dataset .dataloaders() # train_info
            _, test_loader = dataset .dataloaders()
            model = CSWin_tiny(args.numofcat, weighted=False, path=None, fixed=False)
            model.load_state_dict(torch.load(args.path, map_location=torch.device('cpu'))['state_dict'])
            model.to(args.device)
            metrics =  eval(model, test_loader, args.device, vis=args.vis, vis_folder=vis_folder, numofcat=args.numofcat)
            print(metrics)
            with open(result_dir + f'metrics{atepoch}_{args.dataset}.json', "w") as outfile:
                json.dump(metrics, outfile)

    elif args.phase == 'grad_cam':
        # a = torch.rand((2, 3, 224, 224)).to('cuda')
        # b = torch.randint(0, 1, (2,)).to('cuda')
 
        # print(model(a, b))
        model = CSWin_tiny(args.numofcat, weighted=False, path=None, fixed=False)
        print(model)
        target_layer = [model.model.stage4[0].norm2]
        # print(targetlayer)
        model.load_state_dict(torch.load(args.path, map_location=torch.device('cpu'))['state_dict'])
        model.to(args.device)
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            #        like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        from pytorch_grad_cam import GradCAMPlusPlus
        cam = GradCAMPlusPlus(model=model, 
            target_layer=target_layer, 
            reshape_transform=reshape_transform,
            aug_smooth=True,
            eigen_smooth=True
        )
        dataset = ABAP(args.dataset, DATA_PATH, trainsize=1, testsize=1)
        dataset.prepare_data()
        _, test_loader = dataset .dataloaders()
        gradcamplspls(test_loader, cam)

    elif args.phase == 'test':
        test_folder = './test/'
            
        dataset = ABAP(args.dataset, DATA_PATH, trainsize=1, testsize=1)
        dataset.prepare_data()
        # test_loader, _ = dataset .dataloaders() # train_info
        _, test_loader = dataset .dataloaders()
        model = CSWin_tiny(args.numofcat, weighted=False, path=None, fixed=False)
        model.load_state_dict(torch.load(args.path, map_location=torch.device('cpu'))['state_dict'])
        model.to(args.device)
        test(model, test_loader, device=args.device, test_folder='./test/', numofcat=args.numofcat)


if __name__ == '__main__':
    main()


