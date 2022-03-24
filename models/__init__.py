import torchvision
from models import simnet, alexnet, alexnetv2, alexnetv3, vgg, resnet, \
    senet, resnext, densenet, simplenetv1, \
    efficientnetv2, googlenet, xception, mobilenetv2, \
    inceptionv3, wideresnet, shufflenetv2, squeezenet, mnasnet, resnetv2,\
    vggv2


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 40)
    print('LOAD MODEL:', model_name)
    print('-' * 40)

    model = None
    if model_name == 'simnet':
        model = simnet.simnet()
    elif model_name == 'alexnet':
        model = alexnet.alexnet(in_channels, num_classes)
    elif model_name == 'alexnetv2':
        model = alexnetv2.alexnet(in_channels, num_classes)
    elif model_name == 'alexnetv3':
        model = alexnetv3.alexnet(in_channels, num_classes)
    elif model_name == 'vgg16':
        model = vggv2.vgg16_bn(in_channels, num_classes)
    elif model_name == 'resnet18':
        model = resnet.resnet18(in_channels, num_classes)
    elif model_name == 'resnet32':
        model = resnetv2.resnet32(in_channels, num_classes)
    elif model_name == 'resnet34':
        model = resnet.resnet34(in_channels, num_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    elif model_name == 'resnet152':
        model = resnet.resnet152(in_channels, num_classes)
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
            -2: model.features[10], # CONV
            -1: model.classifier[4], # FC
        }
    elif model_name == 'alexnetv2':
        module_modules = {
            0: model.features[0],  # 64*11*11
            1: model.features[3],  # 192*5*5
            2: model.features[6],  # 384*3*3
            3: model.features[8],  # 256*3*3
            4: model.features[10], # 256*3*3
            # -1: model.classifier[4], # FC
        }
    elif model_name == 'alexnetv3':
        module_modules = {
            -2: model.features[10], # CONV
            -1: model.classifier[4], # FC
        }
    elif model_name == 'vgg16':
        module_modules = {
            0: model.features[3],  # 64, 224, 224
            1: model.features[10],  # 128, 112, 112
            2: model.features[20],  # 256, 56, 56
            3: model.features[30],  # 512, 28, 28
            4: model.features[34],  # 512, 14, 14
            5: model.features[37],  # 512, 14, 14
            -2: model.features[40],  # 512, 14, 14, CONV
            -1: model.classifier[3], # FC
        }
    elif model_name == 'resnet18':
        module_modules = {
            1: model.conv2_x[0].residual_function[0],
            2: model.conv2_x[0].residual_function[3],
            3: model.conv2_x[1].residual_function[0],
            4: model.conv2_x[1].residual_function[3],

            5: model.conv3_x[0].residual_function[0],
            6: model.conv3_x[0].residual_function[3],
            7: model.conv3_x[1].residual_function[0],
            8: model.conv3_x[1].residual_function[3],

            9: model.conv4_x[0].residual_function[0],
            10: model.conv4_x[0].residual_function[3],
            11: model.conv4_x[1].residual_function[0],
            12: model.conv4_x[1].residual_function[3],

            13: model.conv5_x[0].residual_function[0],
            14: model.conv5_x[0].residual_function[3],
            15: model.conv5_x[1].residual_function[0],
            16: model.conv5_x[1].residual_function[3],
        }
    elif model_name == 'resnet32':
        module_modules = {
            0: model.layer1[0].conv1,
            1: model.layer1[0].conv2,
            2: model.layer1[1].conv1,
            3: model.layer1[1].conv2,
            4: model.layer1[2].conv1,
            5: model.layer1[2].conv2,
            6: model.layer1[3].conv1,
            7: model.layer1[3].conv2,
            8: model.layer1[4].conv1,
            9: model.layer1[4].conv2,

            10: model.layer2[0].conv1,
            11: model.layer2[0].conv2,
            12: model.layer2[1].conv1,
            13: model.layer2[1].conv2,
            14: model.layer2[2].conv1,
            15: model.layer2[2].conv2,
            16: model.layer2[3].conv1,
            17: model.layer2[3].conv2,
            18: model.layer2[4].conv1,
            19: model.layer2[4].conv2,

            20: model.layer3[0].conv1,
            21: model.layer3[0].conv2,
            22: model.layer3[1].conv1,
            23: model.layer3[1].conv2,
            24: model.layer3[2].conv1,
            25: model.layer3[2].conv2,
            26: model.layer3[3].conv1,
            27: model.layer3[3].conv2,
            28: model.layer3[4].conv1,
            29: model.layer3[4].conv2,
            # -1: model.layer3[4].conv2,
        }
    elif model_name == 'resnet34':
        module_modules = {
            -1: model.conv5_x[2].residual_function[3]  # 512,4,4
        }
    elif model_name == 'resnet50':
        module_modules = {
            # 0: model.conv1[0],  # 64

            1: model.conv2_x[0].residual_function[0],
            2: model.conv2_x[0].residual_function[3],
            3: model.conv2_x[0].residual_function[6],
            4: model.conv2_x[1].residual_function[0],
            5: model.conv2_x[1].residual_function[3],
            6: model.conv2_x[1].residual_function[6],
            7: model.conv2_x[2].residual_function[0],
            8: model.conv2_x[2].residual_function[3],
            9: model.conv2_x[2].residual_function[6],

            10: model.conv3_x[0].residual_function[0],
            11: model.conv3_x[0].residual_function[3],
            12: model.conv3_x[0].residual_function[6],
            13: model.conv3_x[1].residual_function[0],
            14: model.conv3_x[1].residual_function[3],
            15: model.conv3_x[1].residual_function[6],
            16: model.conv3_x[2].residual_function[0],
            17: model.conv3_x[2].residual_function[3],
            18: model.conv3_x[2].residual_function[6],
            19: model.conv3_x[3].residual_function[0],
            20: model.conv3_x[3].residual_function[3],
            21: model.conv3_x[3].residual_function[6],

            22: model.conv4_x[0].residual_function[0],
            23: model.conv4_x[0].residual_function[3],
            24: model.conv4_x[0].residual_function[6],
            25: model.conv4_x[1].residual_function[0],
            26: model.conv4_x[1].residual_function[3],
            27: model.conv4_x[1].residual_function[6],
            28: model.conv4_x[2].residual_function[0],
            29: model.conv4_x[2].residual_function[3],
            30: model.conv4_x[2].residual_function[6],
            31: model.conv4_x[3].residual_function[0],
            32: model.conv4_x[3].residual_function[3],
            33: model.conv4_x[3].residual_function[6],
            34: model.conv4_x[4].residual_function[0],
            35: model.conv4_x[4].residual_function[3],
            36: model.conv4_x[4].residual_function[6],
            37: model.conv4_x[5].residual_function[0],
            38: model.conv4_x[5].residual_function[3],
            39: model.conv4_x[5].residual_function[6],

            40: model.conv5_x[0].residual_function[0],
            41: model.conv5_x[0].residual_function[3],
            42: model.conv5_x[0].residual_function[6],
            43: model.conv5_x[1].residual_function[0],
            44: model.conv5_x[1].residual_function[3],
            45: model.conv5_x[1].residual_function[6],
            46: model.conv5_x[2].residual_function[0],
            47: model.conv5_x[2].residual_function[3],
            -1: model.conv5_x[2].residual_function[6],
            # 48: model.conv5_x[2].residual_function[6],
        }
    elif model_name == 'resnet152':
        module_modules = {
            -1: model.conv5_x[2].residual_function[6],
        }
    elif model_name == 'senet34':
        module_modules = {
            -2: model.stage4[2].residual[3],  # CONV, 512,4,4
            -1: model.stage4[2].excitation[2],  # FC, 512, 32
        }
    elif model_name == 'wideresnet28':
        module_modules = {
            -1: model.layer3[3].conv2  # 640,8,8
        }
    elif model_name == 'resnext50':
        module_modules = {
            0: model.conv2[0].split_transforms[0],
            1: model.conv2[0].split_transforms[3],
            2: model.conv2[0].split_transforms[6],
            3: model.conv2[1].split_transforms[0],
            4: model.conv2[1].split_transforms[3],
            5: model.conv2[1].split_transforms[6],
            6: model.conv2[2].split_transforms[0],
            7: model.conv2[2].split_transforms[3],
            8: model.conv2[2].split_transforms[6],

            9:  model.conv3[0].split_transforms[0],
            10: model.conv3[0].split_transforms[3],
            11: model.conv3[0].split_transforms[6],
            12: model.conv3[1].split_transforms[0],
            13: model.conv3[1].split_transforms[3],
            14: model.conv3[1].split_transforms[6],
            15: model.conv3[2].split_transforms[0],
            16: model.conv3[2].split_transforms[3],
            17: model.conv3[2].split_transforms[6],
            18: model.conv3[3].split_transforms[0],
            19: model.conv3[3].split_transforms[3],
            20: model.conv3[3].split_transforms[6],

            21: model.conv4[0].split_transforms[0],
            22: model.conv4[0].split_transforms[3],
            23: model.conv4[0].split_transforms[6],
            24: model.conv4[1].split_transforms[0],
            25: model.conv4[1].split_transforms[3],
            26: model.conv4[1].split_transforms[6],
            27: model.conv4[2].split_transforms[0],
            28: model.conv4[2].split_transforms[3],
            29: model.conv4[2].split_transforms[6],
            30: model.conv4[3].split_transforms[0],
            31: model.conv4[3].split_transforms[3],
            32: model.conv4[3].split_transforms[6],
            33: model.conv4[4].split_transforms[0],
            34: model.conv4[4].split_transforms[3],
            35: model.conv4[4].split_transforms[6],
            36: model.conv4[5].split_transforms[0],
            37: model.conv4[5].split_transforms[3],
            38: model.conv4[5].split_transforms[6],
            
            39: model.conv5[0].split_transforms[0],
            40: model.conv5[0].split_transforms[3],
            41: model.conv5[0].split_transforms[6],
            42: model.conv5[1].split_transforms[0],
            43: model.conv5[1].split_transforms[3],
            44: model.conv5[1].split_transforms[6],
            45: model.conv5[2].split_transforms[0],
            46: model.conv5[2].split_transforms[3],
            47: model.conv5[2].split_transforms[6],

            # -1: model.conv5[2].split_transforms[6]  # 2048,4,4
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

    if model_layers is not None:
        modules = [module_modules[layer] for layer in model_layers]
    else:
        modules = list(module_modules.values())

    print('-' * 40)
    print('Model Layer:', model_layers)
    # print('Model Module:', modules)
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
