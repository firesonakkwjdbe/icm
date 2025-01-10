# icm
icm analytic


这是对IN-CONTEXT-MATTING上下文模型在图像抠图中的应用的复现工程



## Installation
所有文件均上传在master分支，readme文件在main分支里，若你看不到工程请把左上方的main改成master,您可以将该工程文件全部下载下来，然后部署环境，环境follow the stable diffusion 2 ，具体参数在environment.yml中，您可以通过conda命令创建部署代码所需的环境

您需要在hugging face中自行下载stable diffusion 2到本地，并放置在根目录中，

## A Quick Demo
在终端运行以下脚本
python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
    


## Inference
Run the following command to do inference of IndexNet Matting/Deep Matting on the Adobe Image Matting dataset:

    python scripts/demo_indexnet_matting.py
    
    python scripts/demo_deep_matting.py
    
Please note that:
- `DATA_DIR` should be modified to your dataset directory;
- Images used in Deep Matting has been downsampled by 1/2 to enable the GPU inference. To reproduce the full-resolution results, the inference can be executed on CPU, which takes about 2 days.

Here is the results of IndexNet Matting and our reproduced results of Deep Matting on the Adobe Image Dataset:

| Methods | Remark | #Param. | GFLOPs | SAD | MSE | Grad | Conn | Model |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Deep Matting | Paper | -- | -- | 54.6 | 0.017 | 36.7 | 55.3 | -- |
| Deep Matting | Re-implementation | 130.55M | 32.34 | 55.8 | 0.018 | 34.6 | 56.8 | [Google Drive (522MB)](https://drive.google.com/open?id=1Uws86AGkFqV2S7XkNuR8dz5SOttxh7AY) |
| IndexNet Matting | Ours | 8.15M | 6.30 | 45.8 | 0.013 | 25.9 | 43.7 | Included |

* The original paper reported that there were 491 images, but the released dataset only includes 431 images. Among missing images, 38 of them were said double counted, and the other 24 of them were not released. As a result, we at least use 4.87% fewer training data than the original paper. Thus, the small differerce in performance should be normal.
* The evaluation code (Matlab code implemented by the Deep Image Matting's author) placed in the ``./evaluation_code`` folder is used to report the final performance for a fair comparion. We have also implemented a python version. The numerial difference is subtle.


## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}

@article{hao2020indexnet,
  title={Index Networks},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```
