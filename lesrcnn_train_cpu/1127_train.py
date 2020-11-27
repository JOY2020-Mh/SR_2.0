import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from solver_1127_cpu import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'lesrcnn')
    parser.add_argument("--ckpt_name", type=str)
    
    parser.add_argument("--print_interval", type=int, default=1) #original data is 1000
    #parser.add_argument("--train_data_path", type=str, 
    #                    default="dataset/DIV2K_train.h5")
    parser.add_argument("--ckpt_dir", type=str,
                        default="1127_checkpoint")
    parser.add_argument("--sample_dir", type=str,
                        default="sample/")
    parser.add_argument("--GT_dir", type = str, default = '/home/miaohuan/Documents/Visstyle_Dataset_Steven/HR')
    parser.add_argument("--LR_dir", type = str, default = '/home/miaohuan/Documents/Visstyle_Dataset_Steven/LR')
    parser.add_argument("--test_GT_dir", type = str, default = '/home/miaohuan/Documents/Visstyle_Dataset_Steven/HR')
    parser.add_argument("--test_LR_dir", type = str, default = '/home/miaohuan/Documents/Visstyle_Dataset_Steven/LR')
    
    
    
    parser.add_argument("--num_gpu", type=int, default=1)### if it is CPU, need to change
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--pretrained_model", type=str, default = 'lesrcnn_x4.pth')
    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--group", type=int, default=1)
    
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    #parser.add_argument("--loss_fn", type=str, 
                        #choices=["MSE", "L1", "SmoothL1"], default="L1") #tcw201904082029
    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1","SSIM"], default="MSE")
    return parser.parse_args()

def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    solver = Solver(net, cfg)
    #print 'ds'
    solver.fit()
    #print 'sdsddwdew'

if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    main(cfg)
