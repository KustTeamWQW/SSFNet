import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('-root', type=str, default='./data/')
    parser.add_argument('-result_path', type=str, default='./results/')
    parser.add_argument('-dataset', type=str, default='Houston2018',
                        choices=['Houston2018', 'HAIDONG', 'HHK'])

    # learning setting
    parser.add_argument('--epochs', type=int, default=600,
                        help='end epoch for training')
    parser.add_argument('--lr', type=float, default=10e-6, #    7e-5
                        help='learning rate')
                        
    # model setting
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_reslayer', type=int, default=2,
                        help='number of residual layers')
    parser.add_argument('--num_parallel', type=int, default=2,
                        help='number of modalities')
    parser.add_argument('--bn_threshold', type=int, default=0.002)


    # dataset setting
    parser.add_argument('--patch_size', type=int, default=7,
                        help='image patch size')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=int, default=0.05,
                        help='samples for validation')

    args = parser.parse_args()
    return args
 
