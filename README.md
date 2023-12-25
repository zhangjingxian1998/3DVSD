```
conda create -n 3DVSD python=3.8
conda activate 3DVSD
pip install -r requirement
pip install -e py-bottom-up-attention/.
pip install -e language-evaluation/.
python -c "import language_evaluation; language_evaluation.download('coco')"
```

# Train

```
bash VL-T5/scripts/VSD_3D_pretrain_VLT5.sh
bash VL-T5/scripts/VSD_3D_VLT5_vsd2.sh
bash VL-T5/scripts/VSD_3D_VLT5_vsd1.sh
```

## Debug
```
python VL-T5/src/VSD_3D_debug.py
```

# custom
```
python VL-T5/src/VSD_3D_custom_debug.py
```