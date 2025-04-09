import torch 

def is_cuda():
    return torch.cuda.is_available()
if __name__=="__main__":
    x = is_cuda()
    print(x)
