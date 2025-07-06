import numpy as np


def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))
    fn = np.sum((1 - predict) * actual)
    precision = (tp + 0.000001) / (tp + fp + 0.000001)
    recall = (tp + 0.000001) / (tp + fn + 0.000001)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall, tp, tn, fp, fn


#调整之前的预测值为真实值（后处理）
def point_adjust(score, label, thres):
    predict = score >= thres
    predict= predict
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict, actual


def best_f1_without_pointadjust(score, label):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1_1 = 0.0
    max_f1_th_1 = 0.0
    max_pre = 0.0
    max_recall = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        actual = label
        predict = score >= thres
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1_1:
            max_f1_1 = f1
            max_f1_th_1 = thres
            max_pre = precision
            max_recall = recall
    predict = score >=  max_f1_th_1
    return max_f1_1, max_pre, max_recall, predict


def best_f1(score, label):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    pre = 0.0
    rec = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        predict, actual = point_adjust(score, label, thres=thres)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
    predict, actual = point_adjust(score, label, max_f1_th)
    return max_f1, pre, rec, predict, max_f1_th

def best_f1_thres(score, label, thres):
    predict, actual = point_adjust(score, label, thres=thres)
    f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
    return f1, precision, recall, predict, thres

def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos : min(pos + delay + 1, sp)]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos : min(pos + delay + 1, sp)]:
            new_predict[pos:sp] = 1
        else:
            new_predict[pos:sp] = 0
    return new_predict


def delay_f1(score, label, k=7):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    pre = 0.0
    rec = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        predict = score >= thres
        predict = get_range_proba(predict, label, k)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, label)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
        predict = get_range_proba(score >= max_f1_th, label, k)
    return max_f1, pre, rec, predict


def event_adjust(score, label, thres):
    """
    根据阈值对分数进行预测，调整预测事件的边界使其与实际事件更接近。
    """
    predict = score >= thres
    actual = label > 0.1
    adjusted_predict = np.zeros_like(predict, dtype=bool)
    anomaly_state = False

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            # 扩展预测事件的边界向左调整
            for j in range(i, -1, -1):
                if not actual[j]:
                    break
                adjusted_predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_predict[i] = True

    return adjusted_predict, actual


def calc_event_metrics(predict, actual):
    """
    计算事件级别的 True Positive, False Positive, False Negative。
    """
    tp, fp, fn = 0, 0, 0

    # 找到预测和实际事件的边界
    def get_events(binary_array):
        events = []
        start = None
        for i, val in enumerate(binary_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                events.append((start, i - 1))
                start = None
        if start is not None:
            events.append((start, len(binary_array) - 1))
        return events

    predicted_events = get_events(predict)
    actual_events = get_events(actual)

    # 计算事件级别的 TP, FP, FN
    matched = set()
    for p_start, p_end in predicted_events:
        overlap = False
        for a_idx, (a_start, a_end) in enumerate(actual_events):
            if a_idx in matched:
                continue
            if p_start <= a_end and p_end >= a_start:
                overlap = True
                matched.add(a_idx)
                break
        if overlap:
            tp += 1
        else:
            fp += 1

    fn = len(actual_events) - len(matched)

    return tp, fp, fn


def event_f1(score, label):
    """
    计算最佳 Event F1 Score 及相关指标。
    """
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    best_tp, best_fp, best_fn = 0, 0, 0

    for i in range(0, grain + 1):
        thres = (max_th - min_th) / grain * i + min_th
        predict, actual = event_adjust(score, label, thres)
        tp, fp, fn = calc_event_metrics(predict, actual)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            best_tp, best_fp, best_fn = tp, fp, fn

    precision = best_tp / (best_tp + best_fp) if best_tp + best_fp > 0 else 0
    recall = best_tp / (best_tp + best_fn) if best_tp + best_fn > 0 else 0

    return max_f1, precision, recall, predict 