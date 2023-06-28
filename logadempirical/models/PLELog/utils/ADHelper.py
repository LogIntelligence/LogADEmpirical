import torch.nn.functional as F
import torch.nn  as nn
import torch, random
import numpy as np






class AnomalyDetection(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, inputs):
        # if self.use_cuda:
        #     xlen = len(inputs)
        #     for idx in range(xlen):
        #         inputs[idx] = inputs[idx].cuda(self.device)
        tag_logits, src_represents = self.model(inputs)
        # cache
        self.tag_logits = tag_logits
        self.src_represents = src_represents

    def compute_loss(self, true_tags):
        loss = F.cross_entropy(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()
        return tag_correct, b

    def classifier(self, inputs):
        if inputs[0] is not None:
            self.forward(inputs)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags


class AnomalyDetectionBCELoss(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        # if self.use_cuda:
        #     xlen = len(inputs)
        #     for idx in range(xlen):
        #         inputs[idx] = inputs[idx].cuda(self.device)
        tag_logits, src_represents = self.model(inputs)
        # cache
        self.tag_logits = F.softmax(tag_logits, dim=1)
        self.src_represents = src_represents

    def compute_loss(self, true_tags):
        loss = self.loss(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()
        return tag_correct, b

    # def classifier(self, inputs):
    #     if inputs[0] is not None:
    #         self.forward(inputs)
    #     pred_tags = self.tag_logits.detach().max(1)[1].cpu()
    #     tag_logits = self.tag_logits.detach().cpu().tolist()
    #     return pred_tags, tag_logits

    def classifier(self, inputs,vocab, threshold=0.5):
        if inputs[0] is not None:
            self.forward(inputs)
            pred_tags = self.tag_logits.detach().max(1)[1].cpu()
            tag_logits = self.tag_logits.detach().cpu().numpy().tolist()
            pred_tags.zero_()
            anomaly_tag = vocab.tag2id('yes')
            for i, logits in enumerate(tag_logits):
                if logits[anomaly_tag] >= threshold:
                    pred_tags[i] = anomaly_tag
                else:
                    pred_tags[i] = 1 - anomaly_tag
            return pred_tags, tag_logits
        else:
            return None, None
