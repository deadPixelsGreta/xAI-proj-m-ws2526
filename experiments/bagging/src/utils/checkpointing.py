import os
import torch

# from https://gist.github.com/amaarora/a2d88bfa971ce89aa5a13e006a7c94e5

# TODO: integret with wandb; maybe not needed
# if wandb_enabled and WANDB_AVAILABLE:
#                 wandb.run.summary["best_val_accuracy"] = val_acc
#                 wandb.run.summary["best_epoch"] = epoch + 1

class CheckpointSaver:
    def __init__(
        self,
        dirpath,
        num_classes,
        model_name,
        # config=None,
        top_n=1,
        early_stop_thresh=5,
        ):
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        # self.config = dict(config)
        self.model_name = model_name
        self.early_stop_thresh = early_stop_thresh
        self.early_stop = False
        self.top_model_paths = []
        self.best_val_acc = 0.0
        self.num_classes = num_classes
        
    def __call__(self, model, epoch, val_acc, val_loss, optimizer):
        model_path = self.dirpath / f"best_{self.model_name}_epoch{epoch}.pth"
        if val_acc > self.best_val_acc:
            # logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_val_acc = val_acc
            data = {
                "epoch": epoch,
                "model_name": self.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": self.num_classes
            }
            torch.save(data, model_path)
            
            print(f"   New best model saved! (Val Acc: {val_acc:.2f}%)")

            self.top_model_paths.append({'path': model_path, 'score': val_acc, 'epoch': epoch})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=True)
        elif len(self.top_model_paths) > 0 and (epoch - self.top_model_paths[0]['epoch']) > self.early_stop_thresh:
            self.early_stop = True
        if len(self.top_model_paths)>self.top_n:
            self.cleanup()

        return self.early_stop

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        # logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]