import argparse
import os

from batchgenerators.transforms.sample_normalization_transforms import RangeTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
from monai.networks.nets import UNETR
import torch
from torch.utils.data import DataLoader
import tqdm

from models import Segmentation, VAE
from utils import NiiDataset, filedict_from_json, CropResize, TensorBoardWriter, DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument("prefix")
parser.add_argument("--method", default='unet_train')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_epoch", type=int, default=120)
parser.add_argument("--save_epoch", type=int, default=20)
parser.add_argument("--data_index", default='./data/data_index.json')
parser.add_argument("--train_list", default='MSD_train')
parser.add_argument("--val_list", default='MSD_val')
parser.add_argument("--train_data_root", default='./data/nih/msd')
parser.add_argument("--val_data_root", default='./data/nih/msd')
parser.add_argument("--save_root", default='./model')
parser.add_argument("--display_root", default='./tensorboard')
parser.add_argument("--checkpoint_name", default="best_model.ckpt")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--lambda_vae", type=float, default=0.1)
parser.add_argument("--test_only", action='store_true')
parser.add_argument("--load_prefix_seg", default=None)
parser.add_argument("--load_prefix_vae", default=None)
args = parser.parse_args()

prefix = args.prefix
method = args.method
train_batch = args.batch_size
val_batch = 1
max_epoch = args.max_epoch
save_epoch = args.save_epoch
data_index = args.data_index
train_list = args.train_list
val_list = args.val_list
train_data_root = args.train_data_root
val_data_root = args.val_data_root
save_root = args.save_root
save_path = os.path.join(save_root, prefix)
display_path = os.path.join(args.display_root, prefix)
checkpoint_name = args.checkpoint_name
lr = args.lr
weight_decay = args.weight_decay
lambda_vae = args.lambda_vae
test_only = args.test_only
load_prefix_seg = args.load_prefix_seg
load_prefix_vae = args.load_prefix_vae

num_workers = 4
torch.backends.cudnn.benchmark = True

patch_size = [128, 128, 128]
num_class = 2

for path in [save_path, display_path]:
    os.makedirs(path, exist_ok=True)


def main():
    train_data_list = filedict_from_json(data_index, train_list)
    val_data_list = filedict_from_json(data_index, val_list)

    transforms = {
        "train": Compose([
            CropResize(
                data_key="data",
                label_key="label",
                output_size=patch_size
            ),
            RangeTransform(
                data_key="data",
                label_key="label"
            ),
            SpatialTransform(
                patch_size,
                [dis // 2 - 5 for dis in patch_size],
                random_crop=True,
                scale=(0.85, 1.15),
                do_rotation=True,
                angle_x=(-0.2, 0.2),
                angle_y=(-0.2, 0.2),
                angle_z=(-0.2, 0.2),
                data_key="data",
                label_key="label",
            ),
        ]),
        "val": Compose([
            CropResize(
                data_key="data",
                label_key="label",
                output_size=patch_size
            ),
            RangeTransform(
                data_key="data",
                label_key="label"
            ),
            SpatialTransform(
                patch_size,
                do_rotation=False,
                do_scale=False,
                do_elastic_deform=False,
                random_crop=False,
                data_key="data",
                label_key="label"
            ),
        ])
    }

    print("Loading data")

    train_dataset = NiiDataset(train_data_root, train_data_list, transforms["train"])
    val_dataset = NiiDataset(val_data_root, val_data_list, transforms["val"])

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print("Building model")

    if method in ["unet_train"]:
        model = {
            "teacher": {
                "seg": Segmentation(n_channels=1, n_class=num_class, norm_type=1).cuda(),
                "vae": VAE(n_channels=1, n_class=num_class, norm_type=1, dim=128).cuda(),
            },
            "student": {
                "seg": Segmentation(n_channels=1, n_class=num_class, norm_type=1).cuda()
            }
        }
    elif method in ["unetr_train"]:
        model = {
            "teacher": {
                "seg": UNETR(in_channels=1, out_channels=num_class, img_size=(128, 128, 128)).cuda(),
                "vae": VAE(n_channels=1, n_class=num_class, norm_type=1, dim=128).cuda(),
            },
            "student": {
                "seg": UNETR(in_channels=1, out_channels=num_class, img_size=(128, 128, 128)).cuda()
            }
        }
    else:
        raise ValueError("Try a valid method")

    criterion = DiceLoss(num_class=num_class)
    optimizer = torch.optim.SGD(model["student"]["seg"].parameters(),
                                lr=lr, weight_decay=weight_decay, momentum=0.9)

    print("Loading prefix")

    if load_prefix_seg:
        model_path = os.path.join(save_root, load_prefix_seg, checkpoint_name)
        model_state_dict = torch.load(model_path)['model_state_dict']
        model["teacher"]["seg"].load_state_dict(model_state_dict)
        model["student"]["seg"].load_state_dict(model_state_dict)

    if load_prefix_vae:
        model_path = os.path.join(save_root, load_prefix_vae, checkpoint_name)
        model_state_dict = torch.load(model_path)['model_state_dict']
        model["teacher"]["vae"].load_state_dict(model_state_dict)

    for item in model["teacher"].values():
        for param in item.parameters():
            param.requires_grad = False
        item.eval()

    best_result = 0
    saver = TensorBoardWriter(display_path)

    print("Start training")

    for epoch in tqdm.tqdm(range(max_epoch)):
        if not test_only:
            model["student"]["seg"].train()
            total_final_loss = 0.0
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                model["student"]["seg"].train()
                batch["data"] = batch["data"].cuda()
                batch["label"] = batch["label"].cuda()
                batch["label"] = batch["label"].type(torch.cuda.LongTensor)
                one_hot = torch.cuda.FloatTensor(
                    batch["label"].size(0),
                    num_class,
                    batch["label"].size(2),
                    batch["label"].size(3),
                    batch["label"].size(4)
                ).zero_()
                batch["label"] = one_hot.scatter_(1, batch["label"].data, 1)

                pseudo_label = model["teacher"]["seg"](batch["data"])
                predict_mask = model["student"]["seg"](batch["data"])
                restruct_predict_mask, _, _ = model["teacher"]["vae"](predict_mask)

                recon_loss = criterion(restruct_predict_mask, predict_mask)
                pseudo_loss = criterion(pseudo_label, predict_mask)
                final_loss = lambda_vae * recon_loss + pseudo_loss
                total_final_loss += final_loss

                print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                      (epoch + 1, idx + 1, final_loss, recon_loss, pseudo_loss))

                final_loss.backward()
                optimizer.step()

            saver.add_scale("train_loss", total_final_loss, epoch)

        print("Start validation")

        model["student"]["seg"].eval()
        current_result = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                batch["data"] = batch["data"].cuda()
                batch["label"] = batch["label"].cuda()
                batch["label"] = batch["label"].type(torch.cuda.LongTensor)
                one_hot = torch.cuda.FloatTensor(
                    batch["label"].size(0),
                    num_class,
                    batch["label"].size(2),
                    batch["label"].size(3),
                    batch["label"].size(4)
                ).zero_()
                batch["label"] = one_hot.scatter_(1, batch["label"].data, 1)

                pseudo_label = model["teacher"]["seg"](batch["data"])
                predict_mask = model["student"]["seg"](batch["data"])
                restruct_predict_mask, _, _ = model["teacher"]["vae"](predict_mask, if_random=False, scale=0)

                dice_loss = criterion(predict_mask, batch["label"])
                current_result += dice_loss

                h = predict_mask.shape[4] // 2
                saver.add_image(
                    "val_display",
                    torch.cat(
                        (
                            batch["data"][0:1, 0:1, :, :, h],
                            batch["label"][0:1, 1:2, :, :, h],
                            predict_mask[0:1, 1:2, :, :, h],
                            pseudo_label[0:1, 1:2, :, :, h],
                            restruct_predict_mask[0:1, 1:2, :, :, h],
                        ),
                        dim=0
                    ),
                    idx + epoch * (len(batch))
                )

            current_result = 1 - current_result / (len(val_loader))

            saver.add_scale("val_score", current_result, epoch)
            print('epoch %d validation result: %f, best result %f.' %
                  (epoch + 1, current_result, best_result))

        if test_only:
            break

        if (epoch + 1) % save_epoch == 0:
            print('Saving model')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model["student"]["seg"].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                os.path.join(save_path, f'model_epoch{epoch + 1}.ckpt')
            )
            if current_result > best_result:
                best_result = current_result
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': model["student"]["seg"].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    os.path.join(save_path, 'best_model.ckpt')
                )

    print('Finished Training')


if __name__ == "__main__":
    main()
