from models import simnet, alexnet, vgg, resnet, \
    senet, resnext, densenet, simplenetv1, \
    efficientnetv2, googlenet, xception, mobilenetv2, \
    inceptionv3, wideresnet, shufflenetv2, squeezenet, mnasnet


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 40)
    print('LOAD MODEL:', model_name)
    print('-' * 40)

    model = None
    if model_name == 'simnet':
        model = simnet.simnet()
    elif model_name == 'alexnet':
        model = alexnet.alexnet(in_channels, num_classes)
    elif model_name == 'vgg16':
        model = vgg.vgg16_bn(in_channels, num_classes)
    elif model_name == 'resnet34':
        model = resnet.resnet34(in_channels, num_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    elif model_name == 'senet34':
        model = senet.seresnet34(in_channels, num_classes)
    elif model_name == 'wideresnet28':
        model = wideresnet.wide_resnet28_10(in_channels, num_classes)
    elif model_name == 'resnext50':
        model = resnext.resnext50(in_channels, num_classes)
    elif model_name == 'densenet121':
        model = densenet.densenet121(in_channels, num_classes)
    elif model_name == 'simplenetv1':
        model = simplenetv1.simplenet(in_channels, num_classes)
    elif model_name == 'efficientnetv2s':
        model = efficientnetv2.effnetv2_s(in_channels, num_classes)
    elif model_name == 'efficientnetv2l':
        model = efficientnetv2.effnetv2_l(in_channels, num_classes)
    elif model_name == 'googlenet':
        model = googlenet.googlenet(in_channels, num_classes)
    elif model_name == 'xception':
        model = xception.xception(in_channels, num_classes)
    elif model_name == 'mobilenetv2':
        model = mobilenetv2.mobilenetv2(in_channels, num_classes)
    elif model_name == 'inceptionv3':
        model = inceptionv3.inceptionv3(in_channels, num_classes)
    elif model_name == 'shufflenetv2':
        model = shufflenetv2.shufflenetv2(in_channels, num_classes)
    elif model_name == 'squeezenet':
        model = squeezenet.squeezenet(in_channels, num_classes)
    elif model_name == 'mnasnet':
        model = mnasnet.mnasnet(in_channels, num_classes)
    return model


def load_modules(model, model_name, model_layers):
    module_modules = None
    if model_name == 'alexnet':
        module_modules = {
            -1: model.features[3]  # 256, 4, 4
        }
    elif model_name == 'vgg16':
        module_modules = {
            0: model.features[3],  # 64, 224, 224
            1: model.features[10],  # 128, 112, 112
            2: model.features[20],  # 256, 56, 56
            3: model.features[30],  # 512, 28, 28
            -3: model.features[34],  # 512, 14, 14
            -2: model.features[37],  # 512, 14, 14
            -1: model.features[40],  # 512, 14, 14
        }
    elif model_name == 'resnet34':
        module_modules = {
            -1: model.conv5_x[2].residual_function[3]  # 512,4,4
        }
    elif model_name == 'resnet50':
        module_modules = {
            -1: model.conv5_x[2].residual_function[6]  # 2048,4,4
        }
    elif model_name == 'senet34':
        module_modules = {
            -1: model.stage4[2].residual[3]  # 512,4,4
        }
    elif model_name == 'wideresnet28':
        module_modules = {
            -1: model.layer3[3].conv2  # 640,8,8
        }
    elif model_name == 'resnext50':
        module_modules = {
            -1: model.conv5[2].split_transforms[6]  # 2048,4,4
        }
    elif model_name == 'densenet121':
        module_modules = {
            -1: model.features.dense_block3.bottle_neck_layer_15.bottle_neck[5]  # 32, 4, 4
        }
    elif model_name == 'simplenetv1':
        module_modules = {
            -1: model.features[46]  # 256, 1, 1
        }
    elif model_name == 'efficientnetv2s':
        module_modules = {
            -1: model.features[40].conv[7]  # 256, 1, 1
        }
    elif model_name == 'efficientnetv2l':
        module_modules = {
            -1: model.features[79].conv[7]  # 640, 1, 1
        }
    elif model_name == 'googlenet':
        module_modules = {
            -1: model.b5.b4[1]  # 128, 4, 4
        }
    elif model_name == 'xception':
        module_modules = {
            -1: model.exit_flow.conv[3]  # 2048, 4, 4
        }
    elif model_name == 'mobilenetv2':
        module_modules = {
            -1: model.conv1[0]  # 1280, 5, 5
        }
    elif model_name == 'inceptionv3':
        module_modules = {
            -1: model.Mixed_7c.branch_pool[1].conv  # 192, 6, 6
        }
    elif model_name == 'shufflenetv2':
        module_modules = {
            -1: model.conv5[0]  # 1024, 4, 4
        }
    elif model_name == 'squeezenet':
        module_modules = {
            -1: model.fire9.expand_3x3[0]  # 256, 4, 4
        }
    elif model_name == 'mnasnet':
        module_modules = {
            -1: model.features[18][0]  # 1280, 1, 1
        }

    modules = [module_modules[layer] for layer in model_layers]

    print('-' * 40)
    print('Model Layer:', model_layers)
    print('Model Module:', modules)
    print('-' * 40)

    return modules


if __name__ == '__main__':
    from torchsummary import summary

    # net = load_model('vgg16')
    # mod = load_modules(net, 'vgg16', [-1, -2])
    # print(mod)
    # summary(net, (3, 32, 32))
    # print(net)
    model_list = [
        # 'alexnet',#5 20
        # 'vgg16',#13 51
        # 'resnet50',#53 194
        # 'senet34',  # 36 255
        # 'wideresnet28',#28 93
        # 'resnext50',#53 194
        # 'densenet121',#120 489
        # 'simplenetv1',#13 52
        # 'efficientnetv2s',#112 591
        # 'googlenet',#66 257
        # 'xception',#74 244
        # 'mobilenetv2',#54 184
        # 'inceptionv3',#94 427
        # 'shufflenetv2',#56 202
        # 'squeezenet',#26 112
        'mnasnet'#52 179
    ]
    for mn in model_list:
        print(mn)
        net = load_model(mn, in_channels=1, num_classes=100)
        # summary(net, (1, 32, 32))
        print(net)
