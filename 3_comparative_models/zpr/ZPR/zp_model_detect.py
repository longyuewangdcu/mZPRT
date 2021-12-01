import torch
import torch.nn as nn

from transformers import BertModel,BertPreTrainedModel
import utils
class BertZP(BertPreTrainedModel):
    def __init__(self, config, char2word, pro_num):
        super(BertZP, self).__init__(config)
        assert type(pro_num) is int and pro_num > 1
        self.pro_num = pro_num
        self.char2word = char2word
        self.bert = BertModel(config)

        self.dropout = config.hidden_dropout_prob
        #self.resolution_classifier = SpanClassifier(config.hidden_size, max_relative_position)
        self.detection_classifier = self.build_classifier(config.hidden_size,2)
        self.recovery_classifier = self.build_classifier(config.hidden_size,pro_num)


    def build_classifier(self,input_size,output_size,bias=False):
        m = nn.Linear(input_size,output_size,bias=bias)
        nn.init.xavier_uniform_(m.weight)
        return m

    def forward(self,input_ids,mask,decision_mask,detection_refs,recovery_refs,batch_type):
        char_repre = self.bert(input_ids, attention_mask=mask,token_type_ids=decision_mask)
        char_repre = char_repre.last_hidden_state
        char_repre = nn.functional.dropout(char_repre,0.5,training=self.training)  # [batch, seq, dim]
        # bzs,seq_l,dim = char_repre.size()

        # position prediction
#         detection_logits = self.detection_classifier(char_repre)
#         detection_loss = utils.detection_loss(detection_logits,detection_refs,mask)
        # detection_loss = utils.bin_CEloss(detection_logits,detection_refs,decision_mask)
#         detection_distributions = torch.nn.functional.log_softmax(detection_logits,dim=-1).detach()
#         detection_outputs = detection_distributions.argmax(dim=-1)  # [batch, seq]
        # detection_outputs = torch.sigmoid(detection_logits).gt(0.5).float() # [batch, seq]
        # recovery_classifier
        recovery_logits = self.recovery_classifier(char_repre)
        recovery_loss = utils.recover_loss(recovery_logits,recovery_refs,mask)
        recovery_distributions = torch.nn.functional.softmax(recovery_logits, dim=-1)
        recovery_outputs = recovery_distributions.argmax(dim=-1)
        recovery_clone = recovery_distributions[:,:,1:].sum(-1)
        recovery_distributions = torch.cat([recovery_distributions[:,:,0].unsqueeze(-1),recovery_clone.unsqueeze(-1)],dim=-1)
        detection_outputs = recovery_distributions.argmax(dim=-1)
        recovery_distributions = torch.log(recovery_distributions)
        detection_loss = utils.nlloss(recovery_distributions,detection_refs,mask)
        
 
#         consist = utils.div_loss(recovery_distributions,detection_distributions,mask)
        total_loss = detection_loss + recovery_loss #+ consist
        return {'total_loss': total_loss, 'detection_loss': detection_loss, 'recovery_loss': recovery_loss},{'detection_outputs': detection_outputs, 'recovery_outputs': recovery_outputs}
