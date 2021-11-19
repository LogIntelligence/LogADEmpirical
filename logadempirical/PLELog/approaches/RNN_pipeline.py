import sys

sys.path.extend(["../../", "../", "./"])
import torch.optim.lr_scheduler
from logadempirical.PLELog.utils.ADHelper import *
from logadempirical.PLELog.model.GRU import *
from logadempirical.PLELog.data.DataLoader import *


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


def train_model(data, dev, test_data, anomaly_detection, vocab, config, vec=None, logger=None,
                outputFile=None, threshold=0.5):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, anomaly_detection.model.parameters()), config)
    bestClassifier = None
    global_step = 0
    bestF = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        logger.info('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num = 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            tinst = batch_data_variable_with_tag_logits(onebatch, vocab)
            # 设置模型为训练模式
            anomaly_detection.model.train()
            if anomaly_detection.use_cuda:
                tinst.to_cuda(anomaly_detection.device)

            anomaly_detection.forward(tinst.inputs)
            loss = anomaly_detection.compute_loss(tinst.targets)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            cur_correct, cur_count = anomaly_detection.compute_accuracy(tinst.truth)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num
            end_time = time.time()
            during_time = end_time - start_time
            if batch_iter % 5 == 0:
                logger.info("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                            % (global_step, acc, iter, batch_iter, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, anomaly_detection.model.parameters()), \
                                         max_norm=config.clip)
                optimizer.step()
                anomaly_detection.model.zero_grad()
                global_step += 1
            if dev:
                if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                    logger.info('Testing on dev set.')
                    _, _, f = evaluate(dev, anomaly_detection, config, vocab, logger)
                    logger.info('\n')
                    if f > bestF:
                        logger.info("Exceed best f: history = %.2f, current = %.2f\n" % (bestF, f))
                        bestF = f
                        bestModel = AttGRUModel(vocab, config, vec)
                        bestModel.load_state_dict(anomaly_detection.model.state_dict())
                        bestClassifier = AnomalyDetectionBCELoss(bestModel, vocab)

        logger.info('Training iter %d finished in %.2f.' % (iter, float(time.time() - start_time)))
    logger.info('Training process finished in %.2f, start testing final model on testing set.' % (
        float(time.time() - start_time)))
    final_p, final_r, final_f = evaluate(test_data, anomaly_detection, config, vocab, logger, threshold=threshold)
    if bestClassifier:
        logger.info('Train finished, start testing based on the best model chosen by dev set.\n')
        torch.save(bestClassifier.model.state_dict(), config.save_model_path)
        dev_p, dev_r, dev_f = evaluate(test_data, bestClassifier, config, vocab, logger, outputFile=None,
                                       threshold=threshold)
    else:
        logger.info(
            'No classifier generated during training process due to dev performance or no dev is given. \n '
            'Testing on testing set using model trained till final epoch')
        torch.save(anomaly_detection.model.state_dict(), config.save_model_path)
        dev_p, dev_r, dev_f = 0, 0, 0
    return dev_p, dev_r, dev_f, final_p, final_r, final_f


def evaluate(data, classifier, config, vocab, logger, outputFile=None, threshold=0.5):
    with torch.no_grad():
        start = time.time()
        classifier.model.eval()
        output = None
        globalBatchNum = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        if outputFile:
            output = open(outputFile, 'w', encoding='utf-8')
        tag_correct, tag_total = 0, 0
        for onebatch in data_iter(data, config.test_batch_size, False):
            tinst = batch_data_variable_with_tag_logits(onebatch, vocab)
            if classifier.use_cuda:
                tinst.to_cuda(classifier.device)
            classifier.model.eval()
            pred_tags, tag_logits = classifier.classifier(tinst.inputs, vocab, threshold)
            for inst, bmatch in batch_variable_inst(onebatch, pred_tags, vocab, tag_logits):
                if outputFile: printInstance(output, inst)
                tag_total += 1
                if bmatch:
                    tag_correct += 1
                    if inst.type == 'Normal':
                        TN += 1
                    else:
                        TP += 1
                else:
                    if inst.type == 'Normal':
                        FP += 1
                    else:
                        FN += 1
            globalBatchNum += 1
        logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
        if TP != 0:
            precision = 100 * TP / (TP + FP)
            recall = 100 * TP / (TP + FN)
            f = 2 * precision * recall / (precision + recall)
            end = time.time()
            logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f, time = %.2f'
                        % (TP, (TP + FP), precision, TP, (TP + FN), recall, f, end - start))
            if outputFile:
                output.close()
        else:
            logger.info('Precision is 0 and therefore f is 0')
            precision, recall, f = 0, 0, 0
    return precision, recall, f

def evaluate_online(data, config, vocab, logger, vec={}, outputFile=None, threshold=0.5, id2tem={}, dat="BGL"):
    from logadempirical.PLELog.data.Sample import load_features
    model = AttGRUModel(vocab, config, vec)
    # if config.use_cuda:
    model = model.cuda()
    classifier = AnomalyDetectionBCELoss(model, vocab)
    classifier.model.load_state_dict(torch.load(config.save_model_path))
    abnormal_insts = {}
    with torch.no_grad():
        start = time.time()
        classifier.model.eval()
        output = None
        globalBatchNum = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        if outputFile:
            output = open(outputFile, 'w', encoding='utf-8')
        tag_correct, tag_total = 0, 0
        for onebatch in data_iter(data, config.test_batch_size, False):
            tinst = batch_data_variable_with_tag_logits(onebatch, vocab)
            if classifier.use_cuda:
                tinst.to_cuda(classifier.device)
            classifier.model.eval()
            pred_tags, tag_logits = classifier.classifier(tinst.inputs, vocab, threshold)
            for inst, bmatch in batch_variable_inst(onebatch, pred_tags, vocab, tag_logits):
                if outputFile: printInstance(output, inst)
                tag_total += 1
                if bmatch:
                    tag_correct += 1
                    if inst.type == 'Normal':
                        TN += 1
                    else:
                        TP += 1
                        abnormal_insts["-".join(inst.src_events)] = 1
                else:
                    if inst.type == 'Normal':
                        FP += 1
                        abnormal_insts["-".join(inst.src_events)] = 1
                    else:
                        FN += 1
            globalBatchNum += 1
    test_data = load_features("{}/test.pkl".format(config.save_dir), only_normal=False)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    lead_time = []
    for idx, (line, lbls) in enumerate(test_data):
        line = list(line)
        seq_len = max(10, len(line))
        line = ["padding"] * (seq_len - len(line)) + line
        try:
            sess_label = max(lbls)
        except:
            sess_label = lbls
        if sess_label == 1:
            FN += 1
        else:
            TN += 1
        for i in range(len(line) - 10 + 1):
            if not isinstance(lbls, int):
                label = max(lbls[i: i + 10])
            else:
                label = lbls
            sequential_pattern = line[i:i + 10]
            sequential_pattern = [id2tem[x] for x in sequential_pattern]
            sequential_pattern = "-".join(sequential_pattern)
            if sequential_pattern in abnormal_insts:
                if label == 1:
                    lead_time.append(i + 11)
                    TP += 1
                else:
                    FP += 1
                break
    with open("{0}/plelog-leadtime.txt".format(config.save_dir), mode="w", encoding="utf8") as f:
        [f.write(str(x) + "\n") for x in lead_time]
    TN = TN - FP
    FN = FN - TP
    logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
    if TP + FP != 0:
        precision = 100 * TP / (TP + FP)
        recall = 100 * TP / (TP + FN)
        f = 2 * precision * recall / (precision + recall)
        sp = TN / (TN + FP)
        end = time.time()
        logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f Specificity = %d / %d = %.4f F1 score = %.4f, time = %.2f'
                    % (TP, (TP + FP), precision, TP, (TP + FN), recall, TN, (TN + FP), sp, f, end - start))
        if outputFile:
            output.close()
    else:
        logger.info('Precision is 0 and therefore f is 0')
        precision, recall, sp, f = 0, 0, 0, 0
    return precision, recall, sp, f

