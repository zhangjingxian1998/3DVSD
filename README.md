```
conda create -n 3DVSD python=3.8
conda activate 3DVSD
pip install -r requirement
pip install -e py-bottom-up-attention/.
pip install -e language-evaluation/.
python -c "import language_evaluation; language_evaluation.download('coco')"
```