# FinQA
A Dataset of Numerical Reasoning over Financial Data
Learned from EMNLP 2021 paper: FinQA
<https://arxiv.org/abs/2109.00122>

![alt text](https://github.com/czyssrs/FinQA/blob/main/eg-intro.png?raw=true)


**Requirements**:

- `pytorch 1.7.1`
- `huggingface transformers 4.4.2`

**Dataset**
The dataset is stored as JSON files in the "dataset" folder. Each entry has the following format:

- `pre_text`: Text before the table
- `post_text`: Text after the table
- `table`: The table itself
- `id`: Unique example ID
- `qa`: Contains fields like the question, reasoning program, supporting facts, execution result, and more

In the private test data, only the `question` field is available, with no reference provided.

**Code**

**The Retriever**

1. **Train**: Edit `config.py` to set project and data paths, and define the folder name for saving checkpoints. Run the command:

   ```sh
   sh run.sh
   ```

   Observe the dev performance to select the best checkpoint.

2. **Inference**: Edit `config.py` to set `mode` to "test" and `saved_model_path` to the selected checkpoint. Then run:

   ```sh
   python Test.py
   ```

   To train the generator, ensure retriever results are generated for all training, dev, and test files.

**The Generator**

1. **Convert Retriever Results**: Use `Convert.py` to convert retriever outputs for training. Set paths in the main entry of `Convert.py`.

2. **Train**: Edit parameters in `config.py`, and run:

   ```sh
   sh run.sh
   ```

3. **Inference**: Edit `config.py` to set `mode` to "test" and specify the checkpoint path. Run:

   ```sh
   python Test.py
   ```

**Evaluation Scripts**
Prepare your prediction file as a list of dictionaries containing `id` and `predicted program`. For example:

```json
[
    {
        "id": "example_id",
        "predicted": [
            "subtract(",
            "5829",
            "5735",
            ")",
            "EOF"
        ]
    },
    ...
]
```

Run the evaluation with:

```sh
python evaluate.py your_prediction_file test.json
```


