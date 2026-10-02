import json, math, re
from collections import Counter
from typing import Any, Dict, Iterable, List

def _safe_div(a,b): return a / b if b else 0.0
def classification_metrics(y_true, y_pred, positive="SUSPICIOUS"):
    tp=sum(a==positive and b==positive for a,b in zip(y_true,y_pred)); tn=sum(a!=positive and b!=positive for a,b in zip(y_true,y_pred)); fp=sum(a!=positive and b==positive for a,b in zip(y_true,y_pred)); fn=sum(a==positive and b!=positive for a,b in zip(y_true,y_pred)); n=len(y_true)
    return {"accuracy": _safe_div(tp+tn,n), "precision": _safe_div(tp,tp+fp), "recall": _safe_div(tp,tp+fn), "f1": _safe_div(2*tp,2*tp+fp+fn), "specificity": _safe_div(tn,tn+fp), "false_positive_rate": _safe_div(fp,fp+tn), "false_negative_rate": _safe_div(fn,fn+tp), "confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}

def pattern_metrics(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true)|set(y_pred)); rows={}
    for label in labels:
        tp=sum(a==label and b==label for a,b in zip(y_true,y_pred)); fp=sum(a!=label and b==label for a,b in zip(y_true,y_pred)); fn=sum(a==label and b!=label for a,b in zip(y_true,y_pred)); support=sum(a==label for a in y_true); p=_safe_div(tp,tp+fp); r=_safe_div(tp,tp+fn); rows[label]={"precision":p,"recall":r,"f1":_safe_div(2*p*r,p+r),"support":support}
    return {"per_class": rows, "macro_precision": sum(v["precision"] for v in rows.values())/len(rows) if rows else 0.0, "macro_recall": sum(v["recall"] for v in rows.values())/len(rows) if rows else 0.0, "macro_f1": sum(v["f1"] for v in rows.values())/len(rows) if rows else 0.0, "weighted_f1": _safe_div(sum(v["f1"]*v["support"] for v in rows.values()),len(y_true)), "confusion_matrix": {a:{b:sum(x==a and y==b for x,y in zip(y_true,y_pred)) for b in labels} for a in labels}}

def schema_metrics(reports: Iterable[dict]):
    reports=list(reports); required={"decision","pattern","risk_level","confidence","evidence","recommended_actions","summary"}; valid=0; missing=0; invalid_enum=0
    for r in reports:
        missing += int(bool(required-set(r))); invalid_enum += int(r.get("decision") not in {"SUSPICIOUS","LEGITIMATE"} or not isinstance(r.get("pattern"),str) or r.get("risk_level") not in {"LOW","MEDIUM","HIGH","CRITICAL"}); valid += int(not (required-set(r)) and r.get("decision") in {"SUSPICIOUS","LEGITIMATE"} and r.get("risk_level") in {"LOW","MEDIUM","HIGH","CRITICAL"})
    return {"valid_json_rate": _safe_div(valid,len(reports)), "schema_compliance_rate": _safe_div(valid,len(reports)), "missing_required_field_rate": _safe_div(missing,len(reports)), "invalid_enum_rate": _safe_div(invalid_enum,len(reports))}

def faithfulness_metrics(traces: Iterable[dict]):
    total=supported=0; hallucinations=0
    for trace in traces:
        context=json.dumps(trace.get("context",{}), default=str).lower(); evidence=trace.get("report",{}).get("evidence",[]); total+=len(evidence)
        for item in evidence:
            if str(item).lower() in context: supported+=1
            else: hallucinations+=1
    return {"evidence_faithfulness_rate":_safe_div(supported,total),"unsupported_evidence_rate":_safe_div(hallucinations,total),"hallucination_count":hallucinations,"evidence_items":total}

def agent_metrics(traces):
    traces=list(traces); calls=[c for t in traces for c in t.get("tool_calls",[])]; successes=sum(c.get("success",False) for c in calls); repeats=0
    for t in traces:
        names=[(c.get("name"),json.dumps(c.get("arguments",{}),sort_keys=True)) for c in t.get("tool_calls",[])] ; repeats += len(names)-len(set(names))
    return {"tool_call_success_rate":_safe_div(successes,len(calls)),"failed_tool_calls":len(calls)-successes,"average_tool_calls_per_case":_safe_div(len(calls),len(traces)),"repeated_identical_tool_calls":repeats,"average_unique_tools_used":_safe_div(sum(len(set(c.get("name") for c in t.get("tool_calls",[]))) for t in traces),len(traces)),"max_tool_budget_hit_rate":_safe_div(sum(t.get("tool_budget_hit",False) for t in traces),len(traces)),"investigations_completed_without_tool_errors":_safe_div(sum(all(c.get("success",False) for c in t.get("tool_calls",[])) for t in traces),len(traces))}

def generation_metrics(predictions, references):
    """Optional text metrics; returns a stable shape when NLP extras are absent."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        bleu=sum(sentence_bleu([r.split()], p.split(), weights=(0.25,0.25,0.25,0.25)) for p,r in zip(predictions,references))/max(len(predictions),1)
    except Exception: bleu=None
    rouge_l=[]
    for pred, ref in zip(predictions, references):
        a,b=pred.lower().split(),ref.lower().split(); dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i,x in enumerate(a,1):
            for j,y in enumerate(b,1): dp[i][j]=dp[i-1][j-1]+1 if x==y else max(dp[i-1][j],dp[i][j-1])
        rouge_l.append(_safe_div(dp[-1][-1],len(b)))
    return {"bleu":bleu,"rouge_l":sum(rouge_l)/max(len(rouge_l),1),"bertscore":None,"bertscore_status":"install bert-score for semantic scoring"}

def efficiency_metrics(traces):
    traces=list(traces); lat=[float(t.get("latency_seconds",0)) for t in traces]; out=[int(t.get("output_tokens",0)) for t in traces]
    return {"average_latency_seconds":sum(lat)/max(len(lat),1),"average_tokens":sum(out)/max(len(out),1),"total_output_tokens":sum(out),"throughput_tokens_per_second":_safe_div(sum(out),sum(lat))}
