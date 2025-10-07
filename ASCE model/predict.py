import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet
import warnings
import matplotlib.pyplot as plt
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(  # 图像预处理
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 存放图像的目录
    img_dir = r"S:\pycharm\徐老师论文课程\植物叶片病害识别论文\小麦\小麦数据\mai"
    assert os.path.exists(img_dir), "directory: '{}' does not exist.".format(img_dir)  # 判断目录是否存在

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)  # 验证json文件是否存在

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    # 创建模型
    model = AlexNet(num_classes=len(class_indict)).to(device)
    weights_path = "./ACSE.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True))  # 载入网络模型
    model.eval()  # 进入eval模式，即关闭dropout方法

    # 加载模型权重，使用weights_only=True
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        output_file_path = 'input.txt'
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(img_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)

                    # 忽略 Matplotlib 的用户警告
                    warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

                    # 设置支持中文的字体
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体

                    # 处理图像并进行预测
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)  # 计算softmax
                    predict_cla = torch.argmax(predict).detach().numpy()
                    prob = predict[predict_cla].detach().numpy()

                    if img_name.endswith(('.jpg', '.png', '.jpeg')):  # 可以根据需求调整支持的文件格式

                        # 从文件名中提取真实类别
                        true_class = img_name.split('.')[0]
                        # 查找预测标签
                        predicted_label = class_indict[str(predict_cla)]
                        # 比较预测和真实类别
                        comparison_result = f'真实: {true_class}      预测: {predicted_label}       准确度: {prob:.4f}'
                        # 打印结果
                        print(comparison_result)

                        # print_res = "Pre: {}   Acc: {:.3}".format(class_indict[str(predict_cla)], prob)  # 打印结果
                        # plt.text(50, 150, print_res, fontsize=15, color='red')
                        #
                        # output_file.write(f'Predict: {class_indict[str(predict_cla)]} Accuracy: {predict[predict_cla].detach().numpy():.4f}\n')
                        #
                        # print(print_res)  # 控制台输出结果
                        # # plt.axis('off')  # 关闭坐标轴
                        # plt.show()  # 显示图片


if __name__ == '__main__':
    main()
