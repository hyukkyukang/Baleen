import collections
import unicodedata


# Helper functions
def allin(list1, list2):
    for e in list1:
        if e not in list2:
            return False
    return True

def normalize_text(t):
    if type(t) == int:
        return t
    return unicodedata.normalize('NFD', t)

def normalize_sp(sp):
    normalized_sp = []
    for _sp in sp:
        normalized_sp.append(normalize_text(_sp[0]) + '_' + str(_sp[1]))
    return normalized_sp

def compute_exact_doc(gold_toks, pred_toks):
    gold_toks = list(filter(normalize_text, gold_toks))
    pred_toks = list(filter(normalize_text, pred_toks))
    if len(gold_toks) != len(pred_toks):
        return 0
    for t in gold_toks:
        if t not in pred_toks:
            return 0
    return 1


def compute_f1_doc(gold_toks, pred_toks):
    gold_toks = list(filter(normalize_text, gold_toks))
    pred_toks = list(filter(normalize_text, pred_toks))
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_sent(gold_toks, pred_toks):
    gold_toks = normalize_sp(gold_toks)
    pred_toks = normalize_sp(pred_toks)
    if len(gold_toks) != len(pred_toks):
        return 0
    for t in gold_toks:
        if t not in pred_toks:
            return 0
    return 1

def compute_f1_sent(a_gold, a_pred):
    gold_toks = normalize_sp(a_gold)
    pred_toks = normalize_sp(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Evaluate supporting facts
def compute_exact_sp(gold_sp, pred_sp):
    if len(gold_sp) != len(pred_sp):
        return 0
    for t in gold_sp:
        if t not in pred_sp:
            return 0
    return 1

def compute_f1_sp(gold_sp, pred_sp):
    common = collections.Counter(gold_sp) & collections.Counter(pred_sp)
    num_same = sum(common.values())
    if len(gold_sp) == 0 or len(pred_sp) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_sp == pred_sp)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_sp)
    recall = 1.0 * num_same / len(gold_sp)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores_sp(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = str(example["qid"])
        supporting_facts = list(map(tuple, example["support_facts"]))

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        predicted_sp = list(map(tuple, prediction[0]))
        
        exact_scores[qas_id] = compute_exact_sp(supporting_facts, predicted_sp)
        f1_scores[qas_id] = compute_f1_sp(supporting_facts, predicted_sp)

    return exact_scores, f1_scores

def make_eval_dict_sp(acc, qid_list=None):
    if not qid_list:
        total = len(acc)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc[k] for k in qid_list) / total),
                ("total", total),
            ]
        )

def hover_evaluate_sp(examples, preds):
    acc = get_raw_scores_sp(examples, preds)
    evaluation = list(map(make_eval_dict_sp,acc))
    return evaluation

# Evaluating sentence retrieval
def hover_evaluate_sent(examples, preds):
    acc = get_raw_scores_sent(examples, preds)
    evaluation = make_eval_dict_sent(acc)
    return evaluation

def get_raw_scores_sent(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        supporting_facts = example.supporting_facts

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        predicted_sp = prediction['predicted_sp']
        
        exact_scores[qas_id] = compute_exact_sent(supporting_facts, predicted_sp)
        f1_scores[qas_id] = compute_f1_sent(supporting_facts, predicted_sp)

    return exact_scores, f1_scores

def make_eval_dict_sent(acc, qid_list=None):
    if not qid_list:
        total = len(acc)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc[k] for k in qid_list) / total),
                ("total", total),
            ]
        )
        
# Evaluating document retrieval
def get_raw_scores_doc(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    r5_scores = {}
    r8_scores = {}
    r10_scores = {}
    r20_scores =  {}
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = str(example["qid"])
        supporting_facts = example["support_facts"]
        sp_titles = []
        for e in supporting_facts:
            if e[0] not in sp_titles:
                sp_titles.append(e[0])

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        r5, r8, r10, r20 = 0, 0, 0, 0

        prediction = preds[qas_id]
        sorted_titles = [p[0] for p in prediction[0]] + prediction[1]
        top5, top8, top10, top20 = sorted_titles[:5], sorted_titles[:8], sorted_titles[:10], sorted_titles[:20]
        if allin(sp_titles, top5):
            r5 = 1
        if allin(sp_titles, top8):
            r8 = 1
        if allin(sp_titles, top10):
            r10 = 1
        if allin(sp_titles, top20):
            r20 = 1
        
        r5_scores[qas_id] = r5
        r8_scores[qas_id] = r8
        r10_scores[qas_id] = r10
        r20_scores[qas_id] = r20

        predicted_docs = [p[0] for p in prediction[0]]
        exact_scores[qas_id] = compute_exact_doc(sp_titles, predicted_docs)
        f1_scores[qas_id] = compute_f1_doc(sp_titles, predicted_docs)


    return r5_scores, r8_scores, r10_scores, r20_scores, exact_scores, f1_scores

def make_eval_dict_doc(r5, r8, r10, r20, exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(r5)
        return collections.OrderedDict(
            [
                ("hit5", 100.0 * sum(r5.values()) / total),
                ("hit8", 100.0 * sum(r8.values()) / total),
                ("hit10", 100.0 * sum(r10.values()) / total),
                ("hit20", 100.0 * sum(r20.values()) / total),
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("hit5", 100.0 * sum(r5[k] for k in qid_list) / total),
                ("hit8", 100.0 * sum(r8[k] for k in qid_list) / total),
                ("hit10", 100.0 * sum(r10[k] for k in qid_list) / total),
                ("hit20", 100.0 * sum(r20[k] for k in qid_list) / total),
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )

def hover_evaluate_doc(examples, preds):
    r5, r8, r10, r20, em, f1 = get_raw_scores_doc(examples, preds)
    evaluation = make_eval_dict_doc(r5, r8, r10, r20, em, f1)
    return evaluation

# Claim verification
def get_raw_scores_claim(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    acc_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_label = example.label

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        pred_label = preds[qas_id]['predicted_label']
        acc_scores[qas_id] = int(pred_label == gold_label)
    return acc_scores

def make_eval_dict_claim(acc, qid_list=None):
    if not qid_list:
        total = len(acc)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("acc", 100.0 * sum(acc[k] for k in qid_list) / total),
                ("total", total),
            ]
        )

def hover_evaluate_claim(examples, preds):
    acc = get_raw_scores_claim(examples, preds)
    evaluation = make_eval_dict_claim(acc)
    return evaluation


if __name__ == "__main__":
    pass