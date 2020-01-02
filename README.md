# crnn_seq2seq_ocr_pytorch

This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN and Sequence to sequence model with attention for image-based sequence recognition tasks, such as scene text recognition and OCR.  


This network architecture is implemented from [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915)

![arch.jpg](https://github.com/bai-shang/crnn_seq2seq_ocr.Pytorch/blob/master/data/arch.jpg?)  


***The crnn+ctc ocr can be found here [bai-shang/crnn_ctc_ocr_tf](https://github.com/bai-shang/crnn_ctc_ocr_tf)***


# Dependencies
All dependencies should be installed are as follow:
* Python3.5
* PyTorch
* opencv-python
* numpy
* Pillow

Required packages can be installed with
```bash
pip3 install -r requirements.txt
```

This software only tested in Python3.5!

# Run demo

Asume your current work directory is "crnn_seq2seq_ocr_pytorch"：  

```bash
#cd crnn_seq2seq_ocr.Pytorch
python3 inference.py --img_path ./data/test_img/20439171_260546633.jpg \
    --encoder model/pretrained/encoder.pth --decoder model/pretrained/decoder.pth
```

Result is:  

![20439171_260546633.jpg](https://github.com/bai-shang/crnn_seq2seq_ocr_pytorch/blob/master/data/test_img/20439171_260546633.jpg?raw=true)

predict_string: 于采取的是“激变”的 => predict_probility: 0.860094428062439  



# Train a new model

* Download [Synthetic Chinese String Dataset](https://pan.baidu.com/s/1dFda6R3#list/path=%2F).  

* Create **train_list.txt** and **test_list.txt** as follow format
```
path/to/your/image/50843500_2726670787.jpg 情笼罩在他们满是沧桑
path/to/your/image/57724421_3902051606.jpg 心态的松弛决定了比赛
path/to/your/image/52041437_3766953320.jpg 虾的鲜美自是不可待言
```
You can use the "[data/convert_text_list.py](https://github.com/bai-shang/crnn_seq2seq_ocr_pytorch/blob/master/data/convert_text_list.py)" script to create the two lists or finish it by yourself.
```
cd data
python3 convert_text_list.py SyntheticChineseStringDataset/train.txt > train_list.txt
python3 convert_text_list.py SyntheticChineseStringDataset/test.txt > test_list.txt
```

* Start training
```
#cd crnn_seq2seq_ocr.Pytorch
python3 --train_list train_list.txt --eval_list test_list.txt --model ./model/crnn/ 
``` 
Then the training messages are printed to terminal like
![](https://github.com/bai-shang/crnn_seq2seq_ocr.Pytorch/blob/master/data/start_train.jpg?)


# Reference
* [caffe_ocr](https://github.com/senlinuc/caffe_ocr)

* [PyTorch Tutorials >  Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)



