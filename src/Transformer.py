from transformers import AutoModel, AutoConfig
import torch
import copy

class RedditTransformer(torch.nn.Module):
    def __init__(self, model_name, num_classes, extra_dropout, num_groups, num_aux):
        super(RedditTransformer, self).__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        for layer in self.encoder.encoder.layer:
            layer.attention.self.dropout = torch.nn.Dropout(self.encoder.config.attention_probs_dropout_prob + extra_dropout)
            layer.output.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob + extra_dropout)
        print(self.encoder.config)
        #self.encoder.embeddings.requires_grad = False
        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
            torch.nn.Linear(config.hidden_size, num_classes),
        )
        if num_groups != None:
            self.encoder.encoder = BertEncoder(config, self.encoder.encoder.layer)
            self.encoder.pooler = BertPooler(config, self.encoder.pooler.dense)
            self.aux = True
            self.classification_head_aux = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
                torch.nn.Linear(config.hidden_size, num_groups),
            )
            self.classification_head_group = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
                torch.nn.Linear(config.hidden_size, num_aux),
            )
        else:
            self.aux = False
    def forward(self, batch):
        outputs = self.encoder(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        if self.aux == True:
            features_main = outputs[0][0][:, 0, :]
            features_aux = outputs[0][1][:, 0, :]
            features_group = outputs[0][2][:, 0, :]
            logits_main = self.classification_head(features_main)
            logits_aux = self.classification_head_aux(features_aux)
            logits_group = self.classification_head_group(features_group)
            return logits_main, logits_aux, logits_group, outputs
        else:
            features = outputs[0][:, 0, :]
            logits_main = self.classification_head(features)
            return logits_main, None, outputs

class BertEncoder(torch.nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        self.layer = layers[:-1] #nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_left = copy.deepcopy(layers[-1])
        self.layer_right = copy.deepcopy(layers[-1])
        self.layer_center = copy.deepcopy(layers[-1])
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            hidden_states, all_attentions, all_hidden_states = self.layer_loop(output_hidden_states, all_hidden_states, hidden_states, layer_module, head_mask, i, encoder_hidden_states, encoder_attention_mask, output_attentions, all_attentions, attention_mask)
        hidden_states_left, all_attentions_left, all_hidden_states_left = self.layer_loop(output_hidden_states, all_hidden_states, hidden_states, self.layer_left, head_mask, len(self.layer), encoder_hidden_states, encoder_attention_mask, output_attentions, all_attentions, attention_mask)
        hidden_states_right, all_attentions_right, all_hidden_states_right = self.layer_loop(output_hidden_states, all_hidden_states, hidden_states, self.layer_right, head_mask, len(self.layer), encoder_hidden_states, encoder_attention_mask, output_attentions, all_attentions, attention_mask)
        hidden_states_center, all_attentions_center, all_hidden_states_center = self.layer_loop(output_hidden_states, all_hidden_states, hidden_states, self.layer_center, head_mask, len(self.layer), encoder_hidden_states, encoder_attention_mask, output_attentions, all_attentions, attention_mask)

        if output_hidden_states:
            all_hidden_states_left = all_hidden_states_left + (hidden_states_left,)
            all_hidden_states_right = all_hidden_states_right + (hidden_states_right,)
            all_hidden_states_center = all_hidden_states_center + (hidden_states_center,)

        outputs = ((hidden_states_left, hidden_states_right, hidden_states_center),)
        if output_hidden_states:
            outputs = outputs + ((all_hidden_states_left, all_hidden_states_right, all_hidden_states_center),)
        if output_attentions:
            outputs = outputs + ((all_attentions_left,all_attentions_right, all_attentions_center),)
        return outputs
    def layer_loop(self, output_hidden_states, all_hidden_states, hidden_states, layer_module, head_mask, i, encoder_hidden_states, encoder_attention_mask, output_attentions, all_attentions, attention_mask):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
        return hidden_states, all_attentions, all_hidden_states


class BertPooler(torch.nn.Module):
    def __init__(self, config, dense):
        super().__init__()
        self.dense_left = copy.deepcopy(dense)
        self.dense_right = copy.deepcopy(dense)
        self.dense_center = copy.deepcopy(dense)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor_left = hidden_states[0][:, 0]
        first_token_tensor_right = hidden_states[1][:, 0]
        first_token_tensor_center = hidden_states[2][:, 0]

        pooled_output_left = self.dense_left(first_token_tensor_left)
        pooled_output_left = self.activation(pooled_output_left)
        pooled_output_right = self.dense_right(first_token_tensor_right)
        pooled_output_right = self.activation(pooled_output_right)
        pooled_output_center = self.dense_right(first_token_tensor_center)
        pooled_output_center = self.activation(pooled_output_center)
        return (pooled_output_left, pooled_output_right, pooled_output_center)