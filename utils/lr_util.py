from torch.optim import lr_scheduler



def _adjust_learning_rate(epoch, gamma=0.01):
    """Sets the learning rate"""
    if epoch < 5:
        decay = (epoch + 1) / 5
    elif epoch < 160:
        decay = 1
    elif epoch < 180:
        decay = gamma
    else:
        decay = gamma**2

    return decay


# 前五轮预热，然后在160轮，180轮递减0.01
def adjust_learning_rate(epoch):
    return _adjust_learning_rate(epoch, 0.01)


# 前五轮预热，然后在160轮，180轮递减0.1
def adjust_learning_rate_10x(epoch):
    """Sets the learning rate"""
    return _adjust_learning_rate(epoch, 0.1)


def get_lr_scheduler(optimizer, verbose=False):
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=adjust_learning_rate,
        verbose=verbose
    )

    return scheduler


def get_lr_scheduler_10x(optimizer, verbose=False):
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=adjust_learning_rate_10x,
        verbose=verbose
    )

    return scheduler
