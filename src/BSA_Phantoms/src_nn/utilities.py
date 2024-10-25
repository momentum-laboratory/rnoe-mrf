import torch
import numpy as np
import matplotlib.pyplot as plt

def check_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU found and will be used")
    else:
        print("GPU was not found. Using CPU")
    return device

def normalize_range(original_array, original_min, original_max, new_min, new_max):
    a, b, c, d = original_min, original_max, new_min, new_max
    return (original_array - a) / (b - a) * (d - c) + c

def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    a, b, c, d = original_min, original_max, new_min, new_max
    return (normalized_array - c) / (d - c) * (b - a) + a

def plt_train_loss(epoch, loss_per_epoch):
    plt.figure()
    plt.plot(np.arange(epoch+1), loss_per_epoch)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('MSE Loss', fontsize=20)
    plt.title('Training Loss', fontsize=20)
    plt.show()

def plt_eval_loss(eval_losses):
    plt.figure()
    plt.plot(np.arange(len(eval_losses)) + 1, eval_losses)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Eval MSE Loss', fontsize=20)
    plt.title('Evaluation Loss', fontsize=20)
    plt.show()