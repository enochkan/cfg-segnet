import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torchvision.transforms as transforms

def main(config, fold):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)

    transforms_ = [
        transforms.ToTensor(),
        # transforms.Normalize([0.5],[0.5]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            transforms=transforms_)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='val',
                            transforms=transforms_)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=2,
                            num_workers=config.num_workers,
                            mode='test',
                            transforms=transforms_,
                            shuffle=False)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train(fold)
    elif config.mode == 'test':
        solver.test('./models/fold4/32.pkl', 'prostate', fold)


if __name__ == '__main__':
    fold = 'fold4'
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='../../data/prostate/'+fold)
    parser.add_argument('--valid_path', type=str, default='../../data/prostate/'+fold)
    parser.add_argument('--test_path', type=str, default='../../data/prostate/'+fold)
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config, fold)
