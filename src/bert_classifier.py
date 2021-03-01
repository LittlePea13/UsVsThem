# -*- coding: utf-8 -*-
from collections import OrderedDict
import csv
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import confusion_matrix, jaccard_score
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from Transformer import RedditTransformer
from dataloader import sentiment_analysis_dataset, MyCollator
import seaborn as sn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import argparse
from sklearn.utils import class_weight
class BERTClassifier(pl.LightningModule):
    """
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser) -> None:
        super(BERTClassifier, self).__init__()
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs
        self.model_name = hparams.encoder_model
        self.prepare_sample = MyCollator(self.model_name, hparams.max_length)

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        '''self.bert = AutoModel.from_pretrained(
            self.hparams.encoder_model
        )'''
        self.hparams.le = LabelEncoder()

        comments = pd.concat([pd.read_csv(self.hparams.train_csv),pd.read_csv(self.hparams.dev_csv), pd.read_csv(self.hparams.test_csv)])
        comments = comments[~comments['is_Disc_Crit'].isna()]
        comments['raw_label'] = comments['is_Disc_Crit'].apply(lambda x: 'Against' if x == True else 'For')
        self.hparams.le.fit(comments['raw_label'])
        comments['label'] = self.hparams.le.transform(comments['raw_label'])
        self.hparams.le_aux = LabelEncoder()
        # self.class_weights = torch.tensor(class_weight.compute_class_weight('balanced',
        #                                                 np.unique(comments['Binary_populism']),
        #                                                 comments['Binary_populism']), dtype=torch.float)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            #self.hparams.aux_task = 'group'
            self.hparams.le_aux.fit(comments[self.hparams.aux_task].values)
            self.model = RedditTransformer(self.hparams.encoder_model, len(self.hparams.le.classes_), self.hparams.extra_dropout, len(self.hparams.le_aux.classes_))
            self.weights = nn.Parameter(torch.Tensor([1 + self.hparams.loss_aux, 1 - self.hparams.loss_aux]), requires_grad=True)
            self.alpha = 0.5
        elif self.hparams.aux_task == 'emotions':
            self.model = RedditTransformer(self.hparams.encoder_model, len(self.hparams.le.classes_), self.hparams.extra_dropout, 8)
            self.weights = nn.Parameter(torch.Tensor([1 + self.hparams.loss_aux, 1 - self.hparams.loss_aux]), requires_grad=True)
            self.alpha = 0.5
        else:
            self.hparams.le_aux = LabelEncoder()
            self.model = RedditTransformer(self.hparams.encoder_model, len(self.hparams.le.classes_), self.hparams.extra_dropout, None)

    def __build_loss(self):
        """ Initializes the loss function/s. """
        # self._loss = nn.CrossEntropyLoss(self.class_weights)
        self._loss = nn.CrossEntropyLoss()
        if self.hparams.aux_task == 'emotions':
            self._loss_aux = nn.BCEWithLogitsLoss()
        else:    
            self._loss_aux = nn.CrossEntropyLoss()
        self._Gradloss = nn.L1Loss()


    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.encoder.encoder.layer[-1].output.parameters():
            param.requires_grad = True
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input = self.prepare_sample([sample])
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, tokens):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        # print('forward', tokens['input_ids'].shape)
        # print('tree_sizes', tokens['tree_sizes'])
        # print('node_order', tokens['node_order'])
        logits, logits_aux, _ = self.model(tokens)

        return {"logits": logits, "logits_aux": logits_aux}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        # print('targets', targets)
        # print('predictions', predictions["logits"].shape)
        loss = self._loss(predictions["logits"], targets["labels"])
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task == 'emotions'):
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"])
            return loss, loss_aux#*self.hparams.loss_aux
        elif (self.hparams.aux_task != 'None'):
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"].type(torch.long))
            return loss, loss_aux  
        return loss, None

    def backward(self, use_amp, loss, optimizer, idx_opt):
        if self.hparams.aux_task != 'None' and self.hparams.gradnorm == True:
            loss_val = self.weights * loss
            total_weighted_loss = loss_val.sum()
            total_weighted_loss.backward(retain_graph = True)
            self.weights.grad = 0.0 * self.weights.grad
            W = list(self.model.encoder.encoder.layer[-1].output.parameters())
            norms = []
            for w_i, L_i in zip(self.weights, loss.flatten()):
                # gradient of L_i(t) w.r.t. W
                gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
                
                # G^{(i)}_W(t)
                norms.append(torch.norm(w_i * gLgW[0]))
            norms = torch.stack(norms)
            if self.trainer.global_step == 0:
                self.initial_losses = loss.detach()
            with torch.no_grad():
                # loss ratios \curl{L}(t)
                loss_ratios = loss / self.initial_losses
                
                # inverse training rate r(t)
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                
                constant_term = norms.mean() * (inverse_train_rates ** self.alpha)
            # write out the gradnorm loss L_grad and set the weight gradients
            grad_norm_loss = (norms - constant_term).abs().sum()
            self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        elif self.hparams.aux_task != 'None':
            loss_val = self.weights * loss
            total_weighted_loss = loss_val.sum()
            total_weighted_loss.backward()
        else:
            loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, lambda_closure, using_native_amp):
        if self.trainer.use_tpu and XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        elif isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(lambda_closure)
        else:
            optimizer.step()
        if self.hparams.aux_task != 'None':
            with torch.no_grad():
                if epoch > self.hparams.warmup_aux:
                    self.weights.data = nn.Parameter(torch.Tensor([1.99, 0.01]), requires_grad=True).to(self.weights.device)
                normalize_coeff = len(self.weights) / self.weights.sum()
                self.weights.data = self.weights.data * normalize_coeff
        # clear gradients
        optimizer.zero_grad()

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        # print('Training', inputs['input_ids'].shape)
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        if self.hparams.aux_task != 'None':
            task_losses = torch.stack(loss_val)
            loss_val = self.weights * task_losses
            total_weighted_loss = loss_val.sum()
        else:
            total_weighted_loss = loss_val[0]
            task_losses = loss_val[0]
        y = targets["labels"]
        y_hat = model_out["logits"]
        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val[0].device.index)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            total_weighted_loss = total_weighted_loss.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)
            task_losses = task_losses.unsqueeze(0)

        if self.hparams.aux_task != 'None':
            tqdm_dict = {"train_loss": total_weighted_loss, "train_acc": val_acc, "weight1": self.weights[0], "weight2": self.weights[1]}
        else:
            tqdm_dict = {"train_loss": total_weighted_loss, "train_acc": val_acc}
        output = OrderedDict(
            {"loss": task_losses, "progress_bar": tqdm_dict, "log": tqdm_dict, "acc": val_acc}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)[0]

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            # Auxiliary
            y_aux = targets["labels_aux"]
            y_hat_aux = model_out["logits_aux"]

            # acc
            labels_hat_aux = torch.argmax(y_hat_aux, dim=1)
            val_acc_aux = torch.sum(y_aux == labels_hat_aux).item() / (len(y) * 1.0)
            val_acc_aux = torch.tensor(val_acc_aux)
        elif self.hparams.aux_task == 'emotions':
            y_aux = targets["labels_aux"]
            y_hat_aux = model_out["logits_aux"]
            labels_hat_aux = (nn.Sigmoid()(y_hat_aux) > 0.5)
            val_acc_aux = jaccard_score(y_aux.cpu(),labels_hat_aux.cpu(), average='macro')
            val_acc_aux = torch.tensor(val_acc_aux)
        else:
            val_acc_aux = torch.tensor(0, dtype = torch.float)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)
            val_acc_aux = val_acc_aux.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)
            val_acc_aux = val_acc_aux.unsqueeze(0)
        conf_matrix = confusion_matrix(self.hparams.le.inverse_transform(y.cpu().numpy()),self.hparams.le.inverse_transform(labels_hat.cpu().numpy()), labels=self.hparams.le.classes_)
        conf_matrix = torch.tensor(conf_matrix, device = loss_val.device.index)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            conf_matrix_aux = confusion_matrix(self.hparams.le_aux.inverse_transform(y_aux.type(torch.long).cpu().numpy()),self.hparams.le_aux.inverse_transform(labels_hat_aux.cpu().numpy()), labels=self.hparams.le_aux.classes_)
            conf_matrix_aux = torch.tensor(conf_matrix_aux, device = loss_val.device.index)
        elif self.hparams.aux_task == 'emotions':
            conf_matrix_aux = confusion_matrix(y_aux.cpu().numpy().argmax(axis=1), labels_hat_aux.cpu().numpy().argmax(axis=1), labels = np.arange(len(self._dev_dataset.columns)))
            conf_matrix_aux = torch.tensor(conf_matrix_aux, device = loss_val.device.index)
        else:
            conf_matrix_aux = torch.zeros([1,1], device = loss_val.device.index)
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, "conf_matrix": conf_matrix, "conf_matrix_aux": conf_matrix_aux, "val_acc_aux": val_acc_aux})

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the test_end function.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)[0]

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        with open('experiments/predictions/' + str(self.hparams.seed) + '_' + self.hparams.aux_task + '_preds.csv','a') as fd:
            for line in zip(labels_hat.tolist(), y.tolist()):
                fd.write(','.join(str(v) for v in line))
                fd.write('\n')

        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            # Auxiliary
            y_aux = targets["labels_aux"]
            y_hat_aux = model_out["logits_aux"]

            # acc
            labels_hat_aux = torch.argmax(y_hat_aux, dim=1)
            val_acc_aux = torch.sum(y_aux == labels_hat_aux).item() / (len(y) * 1.0)
            val_acc_aux = torch.tensor(val_acc_aux)
        elif self.hparams.aux_task == 'emotions':
            y_aux = targets["labels_aux"]
            y_hat_aux = model_out["logits_aux"]
            labels_hat_aux = (nn.Sigmoid()(y_hat_aux) > 0.5)
            val_acc_aux = jaccard_score(y_aux.cpu(),labels_hat_aux.cpu(), average='macro')
            val_acc_aux = torch.tensor(val_acc_aux)
        else:
            val_acc_aux = torch.tensor(0, dtype = torch.float)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)
            val_acc_aux = val_acc_aux.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)
            val_acc_aux = val_acc_aux.unsqueeze(0)
        conf_matrix = confusion_matrix(self.hparams.le.inverse_transform(y.cpu().numpy()),self.hparams.le.inverse_transform(labels_hat.cpu().numpy()), labels=self.hparams.le.classes_)
        conf_matrix = torch.tensor(conf_matrix, device = loss_val.device.index)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            conf_matrix_aux = confusion_matrix(self.hparams.le_aux.inverse_transform(y_aux.type(torch.long).cpu().numpy()),self.hparams.le_aux.inverse_transform(labels_hat_aux.cpu().numpy()), labels=self.hparams.le_aux.classes_)
            conf_matrix_aux = torch.tensor(conf_matrix_aux, device = loss_val.device.index)
        elif self.hparams.aux_task == 'emotions':
            conf_matrix_aux = confusion_matrix(y_aux.cpu().numpy().argmax(axis=1), labels_hat_aux.cpu().numpy().argmax(axis=1), labels = np.arange(len(self._test_dataset.columns)))
            conf_matrix_aux = torch.tensor(conf_matrix_aux, device = loss_val.device.index)
        else:
            conf_matrix_aux = torch.zeros([1,1], device = loss_val.device.index)
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, "conf_matrix": conf_matrix, "conf_matrix_aux": conf_matrix_aux, "val_acc_aux": val_acc_aux})

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    def train_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "train_loss": val_loss_mean,
        }
        return result


    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        conf_matrix = np.zeros(outputs[0]["conf_matrix"].shape)
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc
            conf_matrix += output["conf_matrix"]

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "conf_matrix": conf_matrix,
        }
        return result
    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        val_acc_aux_mean = 0
        conf_matrix = torch.zeros(int(outputs[0]["conf_matrix"].shape[0]/self.hparams.gpus),outputs[0]["conf_matrix"].shape[1], device = outputs[0]["conf_matrix"].device)
        conf_matrix_aux = torch.zeros(int(outputs[0]["conf_matrix_aux"].shape[0]/self.hparams.gpus),outputs[0]["conf_matrix_aux"].shape[1], device = outputs[0]["conf_matrix_aux"].device)
        for output in outputs:
            val_loss = output["val_loss"]
            val_acc_aux = output["val_acc_aux"]
            conf_matrix_ = output["conf_matrix"]
            conf_matrix_aux_ = output["conf_matrix_aux"]
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                conf_matrix_ = sum(torch.split(conf_matrix_, int(conf_matrix_.shape[0]/self.hparams.gpus)))
                conf_matrix_aux_ = sum(torch.split(conf_matrix_aux_, int(conf_matrix_aux_.shape[0]/self.hparams.gpus)))
                val_loss = torch.mean(val_loss)
                val_acc_aux = torch.mean(val_acc_aux)
            val_loss_mean += val_loss
            val_acc_aux_mean += val_acc_aux


            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)
                # val_loss_mean = torch.mean(val_loss_mean)
            val_acc_mean += val_acc
            conf_matrix += conf_matrix_
            conf_matrix_aux += conf_matrix_aux_

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        val_acc_aux_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, 
            "val_acc": val_acc_mean,
            "val_acc_aux": val_acc_aux_mean

        }
        #log_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean, "conf_matrix": conf_matrix}
        fig = plt.figure(figsize = (10,7))
        ax = plt.axes()
        sn.heatmap(conf_matrix.cpu(), annot=True, ax = ax)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.hparams.le.classes_) 
        ax.yaxis.set_ticklabels(self.hparams.le.classes_)
        self.logger.experiment.log_image('confusion matrix', fig)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes()
            sn.heatmap(conf_matrix_aux.cpu(), annot=True, ax = ax)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self.hparams.le_aux.classes_) 
            ax.yaxis.set_ticklabels(self.hparams.le_aux.classes_)
            self.logger.experiment.log_image('confusion matrix Aux', fig)
        elif self.hparams.aux_task == 'emotions':
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes()
            sn.heatmap(conf_matrix_aux.cpu(), annot=True, ax = ax)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self._dev_dataset.columns)
            ax.yaxis.set_ticklabels(self._dev_dataset.columns)
            self.logger.experiment.log_image('confusion matrix Aux', fig)
        plt.close('all')
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "val_acc": val_acc_mean,
            "val_acc_aux": val_acc_aux_mean,
        }
        return result
    def test_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        val_acc_aux_mean = 0
        conf_matrix = torch.zeros(int(outputs[0]["conf_matrix"].shape[0]),outputs[0]["conf_matrix"].shape[1], device = outputs[0]["conf_matrix"].device)
        conf_matrix_aux = torch.zeros(int(outputs[0]["conf_matrix_aux"].shape[0]),outputs[0]["conf_matrix_aux"].shape[1], device = outputs[0]["conf_matrix_aux"].device)
        for output in outputs:
            val_loss = output["val_loss"]
            val_acc_aux = output["val_acc_aux"]
            conf_matrix_ = output["conf_matrix"]
            conf_matrix_aux_ = output["conf_matrix_aux"]
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                # conf_matrix_ = sum(torch.split(conf_matrix_, int(conf_matrix_.shape[0]/self.hparams.gpus)))
                # if self.hparams.aux_task != 'None':
                #     conf_matrix_aux_ = sum(torch.split(conf_matrix_aux_, int(conf_matrix_aux_.shape[0]/self.hparams.gpus)))
                val_loss = torch.mean(val_loss)
                val_acc_aux = torch.mean(val_acc_aux)
            val_loss_mean += val_loss
            val_acc_aux_mean += val_acc_aux


            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)
                # val_loss_mean = torch.mean(val_loss_mean)
            val_acc_mean += val_acc
            conf_matrix += conf_matrix_
            conf_matrix_aux += conf_matrix_aux_

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        val_acc_aux_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, 
            "val_acc": val_acc_mean,
            "val_acc_aux": val_acc_aux_mean

        }
        #log_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean, "conf_matrix": conf_matrix}
        fig = plt.figure(figsize = (10,7))
        ax = plt.axes()
        sn.heatmap(conf_matrix.cpu(), annot=True, ax = ax)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.hparams.le.classes_) 
        ax.yaxis.set_ticklabels(self.hparams.le.classes_)
        self.logger.experiment.log_image('confusion matrix', fig)
        if (self.hparams.aux_task != 'None') & (self.hparams.aux_task != 'emotions'):
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes()
            sn.heatmap(conf_matrix_aux.cpu(), annot=True, ax = ax)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self.hparams.le_aux.classes_) 
            ax.yaxis.set_ticklabels(self.hparams.le_aux.classes_)
            self.logger.experiment.log_image('confusion matrix Aux', fig)
        elif self.hparams.aux_task == 'emotions':
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes()
            sn.heatmap(conf_matrix_aux.cpu(), annot=True, ax = ax)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self._test_dataset.columns)
            ax.yaxis.set_ticklabels(self._test_dataset.columns)
            self.logger.experiment.log_image('confusion matrix Aux', fig)
        plt.close('all')
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "val_acc": val_acc_mean,
            #"conf_matrix": conf_matrix,
            "val_acc_aux": val_acc_aux_mean,
        }
        return result
    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        if self.hparams.aux_task != 'None' and self.hparams.gradnorm:
            optimizer = optim.Adam([{'params':self.model.parameters(), 'lr':self.hparams.learning_rate}, {'params':self.weights, 'lr': 1e-2}])
        elif self.hparams.aux_task == 'emotions':
            params=[]
            aux_params = []
            for n,p in self.model.named_parameters():
                if any(nd in n for nd in ['classification_head_aux','pooler.dense_right','layer_right']):
                    aux_params.append(p)
                else:
                    params.append(p)
            optimizer = optim.Adam([{'params':params, 'lr':self.hparams.learning_rate}, 
            {'params':aux_params, 'lr': self.hparams.learning_rate *10}])
        else:
            optimizer = optim.Adam([{'params':self.model.parameters(), 'lr':self.hparams.learning_rate}])
        train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(self.hparams.warmup_proportion * train_steps), num_cycles =0.5, num_training_steps=train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.hparams.warmup_proportion * train_steps), num_training_steps=train_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
           self.unfreeze_encoder()
    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return sentiment_analysis_dataset(self.hparams, train, val, test)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--gradnorm",
            default=False,
            type=bool,
            help="Use Gradnorm for MTL.",
        )
        parser.opt_list(
            "--aux_task", 
            default='None',
            tunable=False,
            options=['None','bias','group','emotions'],
            type=str, 
            help="Use online logging for Neptune."
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.opt_list(
            "--learning_rate",
            default=3e-05,
            tunable=False,
            options=[5e-05, 3e-05,1e-05],
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--loss_aux",
            default=0.25,
            tunable=True,
            options=[0.1, 0.5, 0.75, 0.99],
            type=float,
            help="Add dropout to Transformer.",
        )
        parser.opt_list(
            "--warmup_aux",
            default=5,
            tunable=True,
            options=[1,3,5,8,10],
            type=int,
            help="Add warmup scheduled learning.",
        )
        parser.opt_list(
            "--extra_dropout",
            default=0,
            tunable=False,
            options=[0,0.05,0.1,0.15,0.2],
            type=float,
            help="Add dropout to Transformer.",
        )
        parser.opt_list(
            "--warmup_proportion",
            default=0,
            tunable=False,
            options=[0,0.1,0.2,0.3],
            type=float,
            help="Add warmup to Transformer.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=False,
            options=[0, 1, 2, 3, 4, 5],
        )
        parser.add_argument(
            "--max_length",
            default=512,
            type=int,
            help="Max length for text.",
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="pos,neg",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/train_usvsthem.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/valid_usvsthem.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/test_usvsthem.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
