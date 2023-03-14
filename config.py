import argparse


def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--dataset', type=str, default="nasbench101", help='nasbench101||nasbench201||nnlqp')

    # Data Loder
    parser.add_argument('--percent', type=float, default=4236, help='trainings samples, percent or numbers')
    parser.add_argument('--data_path', type=str, default="data/nasbench101/")
    parser.add_argument('--aug_data_path', type=str, default=None, help='Path of augmented training dataset')
    # NNLQP
    parser.add_argument('--override_data', action="store_true")
    parser.add_argument('--test_model_type', type=str, default='resnet18')
    parser.add_argument('--finetuning', type=bool, default=False)

    # Overall
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    
    # Network Settings
    # Encoding
    parser.add_argument('--embed_type', type=str, default='nerf', help='Type of position embedding: nerf|trans')
    parser.add_argument('--depths', nargs='+', type=int, default=[6,1,1,1])
    parser.add_argument('--act_function', type=str, default='relu', help='activation function used in taransformer')
    parser.add_argument('--use_extra_token', action="store_true", help='Whether use a extra token to predict')
    parser.add_argument('--multires_x', type=int, default=32, help='Operations encoding dim = 2*multires_x')
    parser.add_argument('--multires_r', type=int, default=32, help='Source position encoding dim = 2*multires_r')
    parser.add_argument('--multires_p', type=int, default=32, help='Position encoding dim = 2*multires_p')
    parser.add_argument('--graph_d_model', type=int, default=192)
    parser.add_argument('--graph_n_head', type=int, default=6)
    parser.add_argument('--graph_d_ff', type=int, default=768)

    #Head
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--avg_tokens', type=bool, default=False, help='Whether average the tokens of embedding to predict')

    # Device
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=32)

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    # Training Parameters
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=10)  
    parser.add_argument('--save_path', type=str, default='model/')
    parser.add_argument('--save_epoch_freq', type=int, default=1000)
    parser.add_argument('--pretrained_path', type=str, default=None) #test

    #EMA
    parser.add_argument('--model_ema', action="store_true")
    parser.add_argument('--model_ema_decay', type=float, default=0.99)
    parser.add_argument('--model_ema_force_cpu', type=bool, default=True, help='')
    parser.add_argument('--model_ema_eval', type=bool, default=True, help='Using ema to eval during training.')

    # Loss Parameters
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='weight of mse loss')
    parser.add_argument('--lambda_diff', type=float, default=0.1, help='weight of diff loss')
    parser.add_argument('--lambda_consistency', type=float, default=0, help='weight of consistency loss')
    parser.add_argument('--rankloss_type', type=str, default='l1', help='Type of position embedding: L1|L2|KLDiv')

    args = parser.parse_args()

    return args