# icm
icm analytic


这是对IN-CONTEXT-MATTING上下文模型在图像抠图中的应用的复现工程



## Installation
所有文件均上传在master分支，readme文件在main分支里，若你看不到工程请把左上方的main改成master,您可以将该工程文件全部下载下来，然后部署环境，环境follow the stable diffusion 2 ，具体参数在environment.yml中，您可以通过conda命令创建部署代码所需的环境

您需要在hugging face中自行下载stable diffusion 2到本地，并放置在根目录中，

## A Quick Demo
在终端运行以下脚本
python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
    




## output
成功运行后会输出测试集的alpha蒙版，您可以在根目录的mse.py,sad.py,GRad.py,CONN.py，分别对应着MSE,SAD,GRAD,CONN性能分析指标，运行这些脚本，可得到该指标下的数据表现，注意需要修改各指标.py中对应的result文件存放路径
