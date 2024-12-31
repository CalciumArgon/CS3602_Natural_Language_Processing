# Exploring Improvements in Chinese Semantic Slot Filling: From Data and Model Perspectives

Final project for CS3602-Natural-Language-Processing, detail requirement is described in `/README_LM.md`. Final report is at `/CS3602_NLP_Project.pdf`.

## Training and Model Setting

1. Tagging Baseline Pipeline

    ```bash
    python scripts/slu_baseline.py --device 0 --name tag_basline --encoder_cell LSTM --decoder_cell FNN
    ```

2. BERT-Based Pipeline

    ```bash
    python scripts/bert_baseline.py --device 0 --name bert_rnn --encoder_cell bert --decoder_cell RNN
    ```

3. Other Implemented Methods
   
    * `--replace_place_name` refers to using augmented data created by replacing `location` slots.
        ```bash
        python scripts/bert_baseline.py --device 0 --name bert_rnn_aug --encoder_cell bert --decoder_cell RNN --replace_place_name
        ```
    * `--fuse_chunk` refers to using Jieba to segment the input sentence and concatenating the chunk-level embedding with word-embedding.
        ```bash
        python scripts/bert_baseline.py --device 0 --name bert_rnn_fuse --encoder_cell bert --decoder_cell RNN --fuse_chunk
        ```
    * `--encoder_cell` in BERT-based pipeline can be chosen from `bert`, `macbert`, `robert`.

## Testing and Inference

Use `--ckpt` to specify the checkpoint path, and add `--testing` to test the model.

For example, if we test our final proposed model:
```bash
python scripts/bert_baseline.py --device 0 --name bert_rnn --encoder_cell bert --decoder_cell RNN --ckpt ./Bert-RNN-augmentation.bin --testing
```

In some situation, such as testing the model that just trained, we don't need to assign checkpoint's path. After trainging process, the model will be saved at `/checkpoints/{name}/{encoder_cell}-{decoder_cell}-lock{lock_bert_ratio}-replace{replace_place_name}.bin`. Therefore to load the model, no need for adding `--ckpt`, just run the **exact same parameters as training, and add `--testing` in the end**.

For example, if we test the BERT baseline:
```bash
python scripts/bert_baseline.py --device 0 --name bert_rnn --encoder_cell bert --decoder_cell RNN --testing
```

## Model

Our final proposed models are based on BERT model, with additional data processing and structure modification. Two of the best performance models with our methods are as follows:

 * **Fusing Word-Chunk Level Embedding**: Acc/P/R/F1: 78.88/82.22/81.02/81.62
    ```
    https://jbox.sjtu.edu.cn/l/dHHr6R
    ```
 * **Data Augmentation**: Acc/P/R/F1: 79.66/83.46/82.06/82.75
    ```
    https://jbox.sjtu.edu.cn/l/a136A1
    ```

## Methodology

Details can be found in `/CS3602_NLP_Project.pdf`


<div style="text-align: center;">
    <img src="https://i.postimg.cc/c44DZnF8/pipeline.png" alt="pipeline illustration" style="width: 90%;">
</div>

* **Data Processing Techniques**: We explore several data processing techniques to improve performance, including augmenting data by randomly replacing slots value, cleaning data, merging dialogue context. Results illustrate that data augmentation with replacing `location` value is highly effective.

<!-- The preliminary study results are shown below: -->

<!-- <div style="text-align: center;">
    <img src="https://i.postimg.cc/FKtgfdVp/preliminary-study.png" alt="pipeline illustration" style="width: 30%;">
</div> -->

* **Fusing Word-Chunk Level Embedding**: BERT model only takes single-word as input, which only captures word-level information. We propose a new pipeline, which segments the input sentence into word-chunk, and concatenates the chunk-level embedding with word-embedding. The pipeline is shown below:

## Results
   
<div style="text-align: center;">
    <img src="https://i.postimg.cc/ncjDvVzP/results-in-readme.png" alt="pipeline illustration" style="width: 90%;">
</div>

   