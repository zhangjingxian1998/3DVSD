# 环境配置
```
conda create -n 3DVSD python=3.8
conda activate 3DVSD
pip install -r requirement
pip install -e py-bottom-up-attention/.
pip install -e language-evaluation/.
python -c "import language_evaluation; language_evaluation.download('coco')"
```

# 数据集
数据集准备来自VSD仓库，下载 VG and SpatialSense 数据集，运行
```
python feature_extraction/sp_proposal.py
or 
python feature_extraction/vg_proposal.py
```
结果输出为一个.h5文件，其中包含VSD数据集的数据。为创建3DVSD所需的数据集，需执行以下步骤
```
python Total3DUnderstanding/extract_detection.py    # 将gt框提取出来，用于下一步total3d的输出。提取出的文件存储在Total3DUnderstanding/data_detection文件夹中,为每个图片的.json文件
python Total3DUnderstanding/extral_3d.py            # 调用total3d框架，提取3D特征，中间文件存储在save_mat_gt_path文件夹中
python Total3DUnderstanding/generate_datasets.py    # 读取save_mat_gt_path文件中的3D文件，生成新的.h5文件
```

# 具体步骤
函数的入口在VSD_3D.py或者VSD_3D_debug.py的main函数中
训练过程构建分词器tokenizer，新增特殊token\<TGT>\<REL>\<SEP>\<OBJ>，预训练过程中由于新增了token，加载权重时需要手动对词向量部分的权重手动赋值。然后构建数据集，优化器，模型。训练过程调用trainer.train()，包含迭代训练，训练中的验证以及训练结束后的测试过程。单独使用测试过程需指定超参--test_only，则会调用trainer.test()。
在非开放世界测试下，所有数据已经经过Total3D模型处理，所以输入数据首先送入3DVSD模型进行计算，计算结果包含三维信息与物体关系的隐式向量r_G，以及用于视觉语言模型的提示词。接下来将结果输入视觉语言模型中做最终运算。
# 训练
```
# 对视觉语言模型的预训练，使用预训练权重为对应视觉语言模型的权重文件。输入主语和宾语对应三维框坐标，经过三维空间位置关系判断，得出'<TGT> sub <REL> rel_p <TGT> obj'的视觉语言模型的语言部分输入，监督信号为'sub,rel_t,obj'，可以认为此预训练为训练标志符<TGT>与<REL>以及建立三维空间关系rel_p与二维空间关系rel_t之间的一个映射。
bash VL-T5/scripts/VSD_3D_pretrain_VLT5.sh

# 全局训练，使用预训练权重为上一步的预训练权重
bash VL-T5/scripts/VSD_3D_VLT5_vsd2.sh
bash VL-T5/scripts/VSD_3D_VLT5_vsd1.sh
# 生成的权重文件会保存在VL-T5/snap/VSD_3D中对应.sh脚本的name文件下
```

## Debug
```
python VL-T5/src/VSD_3D_debug.py
```

# 开放世界测试
```
python VL-T5/src/VSD_3D_custom_debug.py
```