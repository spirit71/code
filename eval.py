
import os
import torch
import parser
import logging
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
# args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
# commons.setup_logging(args.save_dir)
# commons.make_deterministic(args.seed)
# 如果命令行传了 --output_dir，就写到那个目录；否则保持原行为
if hasattr(args, "output_dir") and args.output_dir is not None:
    args.save_dir = args.output_dir
else:
    args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

commons.setup_logging(args.save_dir)

# logging.info(f"Arguments: {args}")
# logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.VPRNet()
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)
args.features_dim = 4096
if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
# recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
# logging.info(f"Recalls on {test_ds}: {recalls_str}")

# logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
all_recalls, combined_str = test.test(args, test_ds, model, args.test_method, pca)
for folder_name, (_, recalls_str) in all_recalls.items():
    logging.info(f"Recalls on {test_ds.dataset_name}/{folder_name}: {recalls_str}")
logging.info(f"Combined recalls on {test_ds}: {combined_str}")
import re, json, os
def parse_combined_recalls(s: str):
    # 解析形如 "R@1: 91.509, R@5: 98.002, R@10: 98.975, R@20: 99.540"
    out = {}
    for k, v in re.findall(r"R@(\d+):\s*([0-9]*\.?[0-9]+)", s):
        out[f"R@{k}"] = float(v)
    return out

metrics = parse_combined_recalls(combined_str)

# 建议把 eval 输出写到一个固定位置：args.save_dir/metrics
metrics_dir = os.path.join(args.save_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

# 文件名用 dataset 名，避免覆盖
safe_name = args.eval_dataset_name.replace("/", "_")
out_path = os.path.join(metrics_dir, f"eval_{safe_name}.json")
with open(out_path, "w") as f:
    json.dump(
        {
            "dataset": args.eval_dataset_name,
            "metrics": metrics,
            "combined_str": combined_str,
            "resume": args.resume,
            "seed": args.seed,
        },
        f,
        ensure_ascii=False,
        indent=2
    )
logging.info(f"Saved eval metrics to: {out_path}")
