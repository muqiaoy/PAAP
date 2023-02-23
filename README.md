# PAAPLoss

To be present at ICASSP 2023.

Title: PAAPLoss: A Phonetic-Aligned Acoustic Parameter Loss for Speech Enhancement

Arxiv: https://arxiv.org/abs/2302.08095

## Prerequisites
```
pip install -r requirements.txt
```

## Datasets
1. Please follow https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master to download the DNS Interspeech 2020 dataset.

2. Edit paths in `noisyspeech_synthesizer.cfg` and run `noisyspeech_synthesizer_multiprocessing.py` to generate your train (and validation) data.

    Most likely, you will not want to change the other parameters in .cfg for the train data, and then you will get 12,000 synthesized audios. You may change the `fileindex_end` in the .cfg to have a small set of validation data. 

    You can also manually change `num_train_files` in `conf/` to adjust the number of train audios in use.

3. Edit paths in `conf/` to make it consistent to your folders that contains the data.



## Usage
1. (Optional) Train the Acoustic estimator (or use the pretrained ones).

    Generating the acoustic feature for the first time could be slow and take up some space.
    ```
    python train_est.py estimator=acoustic
    ```
2. Prepare the json list of the train/valid/test data.
    ```
    bash make_dns.sh
    ```

3. Finetune the enhancement model (only support Demucs / FullSubNet so far).
    The pretrained model checkpoints can be downloaded at the original authors' repositories.
    ```
    python train.py finetune=demucs
    ```
    or
    ```
    python train.py finetune=fullsubnet
    ```
    By default it takes up all of the available GPUs.

4. This objective function can also be used at arbitrary model by using the pretrained acoustic estimator.


## Acknowledgement

Some of the model architectures are adapted from the original [Demucs](https://github.com/facebookresearch/denoiser) and [FullSubNet](https://github.com/Audio-WestlakeU/FullSubNet) repos. The phonetic aligner is adapted from [here](https://github.com/lingjzhu/charsiu). Thanks all the authors for open sourcing!

## Citation

Welcome to cite our paper if you find our code or paper useful for your research!

```
@article{yang2023paaploss,
  title={PAAPLoss: A Phonetic-Aligned Acoustic Parameter Loss for Speech Enhancement},
  author={Yang, Muqiao and Konan, Joseph and Bick, David and Zeng, Yunyang and Han, Shuo and Kumar, Anurag and Watanabe, Shinji and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2302.08095},
  year={2023}
}
```
