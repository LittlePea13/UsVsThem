# Us Vs Them Modeling

Main libraries used:
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)
- [Neptune](https://neptune.ai/blog/pytorch-lightning-neptune-integration)
- [PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/index.html)

## Requirements:

Install the requirements (inside the project folder):
```bash
pip install -r requirements.txt
```

### Train Examples:
#### Regression Task MTL emotions:
```bash
python3 training_reg.py --gpus 4 --batch_size 64 --patience 10 --encoder_model roberta-base --max_epochs 20 --aux_task emotions --learning_rate 0.00003 --nr_frozen_epochs 0 --extra_dropout 0.05 --warmup_proportion 0.1 --loss_aux 0.95 --warmup_aux 8 --seed 1 
```
#### Classification Task STL:
```bash
python3 training.py --gpus 4 --batch_size 64 --patience 10 --encoder_model roberta-base --max_epochs 20 --aux_task None --learning_rate 0.00005 --nr_frozen_epochs 0 --extra_dropout 0.2 --warmup_proportion 0.1 --loss_aux 0.75 --warmup_aux 8 --seed 1
```
#### Regression Task three-MTL:
Use the code at folder Three-task:

```bash
python3 training_reg.py --gpus 4 --batch_size 64 --patience 10 --encoder_model roberta-base --max_epochs 20 --aux_task emotions --learning_rate 0.00003 --nr_frozen_epochs 0 --extra_dropout 0.05 --warmup_proportion 0.1 --loss_aux 0.95 --warmup_aux 8 --seed 1
```

### Testing:
```bash
python3 testing.py --checkpoint_path path_to_model_checkpoint
```
The code was based on Huggingface [Transformers](https://huggingface.co/transformers/index.html) and the[lightning-text-classification](https://github.com/ricardorei/lightning-text-classification) repository.

## Dataset

Due to GDPR we provide the Us Vs. Them dataset with just the Reddit comment body.