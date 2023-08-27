import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_anomaly(results, seq_threshold=0.5):
    total_errors = 0
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold:
            total_errors += 1
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, seq_range):
    best_result = [0] * 9
    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results, seq_th)
        TP = compute_anomaly(test_abnormal_results, seq_th)

        if TP == 0:
            continue

        TN = len(test_normal_results) - FP
        FN = len(test_abnormal_results) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        if F1 > best_result[-1]:
            best_result = [0, seq_th, FP, TP, TN, FN, P, R, F1]
    return best_result


def detect_logkey_anomaly(masked_output, masked_label):
    num_undetected_tokens = 0
    for i, token in enumerate(masked_label):
        # output_maskes.append(torch.argsort(-masked_output[i])[:30].cpu().numpy()) # extract top 30 candidates for mask labels

        if token not in torch.argsort(-masked_output[i])[:6]:  # 6 is num_candidates
            num_undetected_tokens += 1

    return num_undetected_tokens, masked_label.cpu().numpy()


def helper(model, test_dataset, device="cpu", batch_size=32):
    # use 1/10 test data
    total_results = []
    # use large batch size in test data
    data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5,
                             collate_fn=test_dataset.collate_fn)

    for idx, batch in enumerate(data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}

        result = model(batch)
        mask_lm_output = result.embeddings

        for i in range(len(batch["label"])):
            seq_results = {"num_error": 0,
                           "undetected_tokens": 0,
                           "masked_tokens": 0,
                           "total_logkey": torch.sum(batch["sequential"][i] > 0).item(),
                           "deepSVDD_label": 0
                           }

            mask_index = batch["label"][i] > 0
            num_masked = torch.sum(mask_index).tolist()
            seq_results["masked_tokens"] = num_masked

            num_undetected, output_seq = detect_logkey_anomaly(
                mask_lm_output[i][mask_index], batch["label"][i][mask_index])
            seq_results["undetected_tokens"] = num_undetected

            total_results.append(seq_results)
    return total_results


def predict(model, normal_dataset, abnormal_dataset, device="cpu"):
    model.eval()
    print("test normal predicting")
    test_normal_results = helper(model, normal_dataset, device=device)

    print("test abnormal predicting")
    test_abnormal_results = helper(model, abnormal_dataset, device=device)
    best_th, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(test_normal_results,
                                                                         test_abnormal_results,
                                                                         seq_range=np.arange(0, 1, 0.1))

    print("best threshold: {}, best threshold ratio: {}".format(best_th, best_seq_th))
    print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
    print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
    acc = (TP + TN) / (TP + TN + FN + FP)
    return acc, P, R, F1
