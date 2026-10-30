import { useEffect, useMemo, useState } from 'react';
import dashboardData from '../data/dashboard.json';
import PageShell, { Panel, Stat } from '../components/PageShell';
import { go, loadCaseId, loadTransaction } from '../state';
import { getLiveGraph, runLiveInvestigation, saveLiveReport } from '../data/live';
import { getReview, saveReview, type AnalystDecision } from '../data/reviews';

export default function LLMPage() {
  const context = loadTransaction();
  const liveMode = Boolean(context?.case_id || context?.source === 'replayed test transaction');
  const initialIndex = Math.max(0, dashboardData.reports.findIndex((item) => item.case === loadCaseId()));
  const [selected, setSelected] = useState(initialIndex);
  const report = dashboardData.reports[selected] ?? dashboardData.reports[0];
  const [liveReport, setLiveReport] = useState<Record<string, unknown> | null>(null);
  const [liveError, setLiveError] = useState('');
  const [running, setRunning] = useState(false);
  const [graphDetails, setGraphDetails] = useState<Record<string, unknown> | null>(null);
  const [analystDecision, setAnalystDecision] = useState<AnalystDecision | ''>('');
  const [analystNote, setAnalystNote] = useState('');
  const [reviewSaved, setReviewSaved] = useState(false);

  useEffect(() => {
    if (!liveMode || !context?.src_acct) return;
    getLiveGraph(String(context.src_acct)).then((result) => setGraphDetails(result.account)).catch(() => setGraphDetails(null));
  }, [context?.src_acct, liveMode]);

  useEffect(() => {
    const saved = getReview(String(loadCaseId() || context?.case_id || report?.case || ''));
    setAnalystDecision(saved?.decision || '');
    setAnalystNote(saved?.note || '');
    setReviewSaved(false);
  }, [context?.case_id, report?.case]);

  const liveEvidence = context ? [
    `XGBoost fraud probability: ${Number(context.xgboost_score ?? 0).toFixed(3)}`,
    `Transaction: ${String(context.src_acct)} → ${String(context.dst_acct)}; amount ${String(context['Amount Paid'])}; format ${String(context['Payment Format'])}.`,
    graphDetails ? `Graph context: ${graphDetails.total_degree ?? 0} total degree (${graphDetails.out_degree ?? 0} outgoing, ${graphDetails.in_degree ?? 0} incoming); centrality ${Number(graphDetails.degree_centrality ?? 0).toFixed(5)}.` : 'Graph context is loading from the live network endpoint.',
  ] : [];

  const shown = useMemo(() => {
    if (!liveMode || !context) return report;
    const score = Number(context.xgboost_score ?? context.fraud_probability ?? 0);
    return {
      case: String(context.case_id ?? 'live-case'), account: String(context.src_acct ?? '—'),
      decision: score >= 0.3 ? 'SUSPICIOUS' : 'LEGITIMATE', pattern: 'PENDING INVESTIGATION',
      riskLevel: score >= 0.7 ? 'HIGH' : score >= 0.3 ? 'MEDIUM' : 'LOW', confidence: score,
      summary: 'This live transaction was escalated from the XGBoost risk score. Run Gemma to generate the case report.', evidence: liveEvidence, actions: ['Run the live Gemma investigation'],
    };
  }, [context, graphDetails, liveMode, report]);

  if (!shown) return <PageShell title="LLM Investigator" subtitle="No investigation context is available." />;

  const investigateLive = async () => {
    setRunning(true); setLiveError('');
    try {
      const liveContext = liveMode && context ? context : null;
      const contextPayload = liveContext ? {
        accounts: [{ account_id: String(liveContext.src_acct ?? '') }, { account_id: String(liveContext.dst_acct ?? '') }],
        transactions: [liveContext], graph_stats: { source: 'live XGBoost replay', xgboost_score: liveContext.xgboost_score, risk_level: liveContext.risk_level, ...(graphDetails || {}) }, retrieved_evidence: [`Transaction selected from live XGBoost queue with score ${Number(liveContext.xgboost_score ?? 0).toFixed(3)}.`],
      } : { accounts: report.account ? [{ account_id: report.account }] : [], transactions: context ? [context] : [], graph_stats: { source: 'Saved benchmark case', pattern_hint: report.pattern }, retrieved_evidence: report.evidence };
      const result = await runLiveInvestigation({ case_id: String(liveContext?.case_id ?? report.case), context: contextPayload });
      setLiveReport({ ...result.report, __report_source: result.report_source });
      if (result.report_source === 'raw_model_json') {
        saveLiveReport({
          reportId: `LIVE-${result.case_id}`,
          case: result.case_id,
          account: String(liveContext?.src_acct ?? report.account ?? '—'),
          decision: String(result.report.decision ?? '—'),
          pattern: String(result.report.pattern ?? '—'),
          riskLevel: String(result.report.risk_level ?? '—'),
          model: 'Gemma 4 31B',
          status: 'Live session report',
          summary: String(result.report.summary ?? ''),
          evidence: Array.isArray(result.report.evidence) ? result.report.evidence.map(String) : [],
          actions: Array.isArray(result.report.recommended_actions) ? result.report.recommended_actions.map(String) : [],
          confidence: Number(result.report.confidence ?? 0),
        });
      }
    } catch (error) { setLiveError(error instanceof Error ? error.message : 'Live Gemma request failed'); }
    finally { setRunning(false); }
  };

  const invalidLiveReport = liveReport?.__report_source === 'agent_parser' && Number(liveReport.confidence || 0) === 0;
  const liveScore = Number(context?.xgboost_score ?? context?.fraud_probability ?? 0);
  const current = liveReport ? { ...shown, decision: liveReport.decision, pattern: liveReport.pattern, riskLevel: liveReport.risk_level, confidence: liveReport.confidence, summary: liveReport.summary || 'No narrative Gemma summary was returned; use the evidence and network context for analyst review.', rationale: liveReport.rationale, rationalePoints: Array.isArray(liveReport.rationale_points) ? liveReport.rationale_points : [], evidence: Array.isArray(liveReport.evidence) && liveReport.evidence.length ? liveReport.evidence : shown.evidence, actions: Array.isArray(liveReport.recommended_actions) && liveReport.recommended_actions.length ? liveReport.recommended_actions : shown.actions } : shown;
  const centered = 'text-center [&>div:first-child]:mx-auto';
  const saveAnalystReview = () => {
    if (!analystDecision) return;
    saveReview({ caseId: String(current.case), decision: analystDecision, note: analystNote.trim(), reviewedAt: new Date().toISOString() });
    setReviewSaved(true);
  };

  return <PageShell title="LLM Investigator" subtitle={liveMode ? 'Investigate the selected suspicious transaction with a fresh Gemma report.' : 'Saved benchmark cases are available for evaluation only.'} action={!liveMode ? <select value={selected} onChange={(e) => setSelected(Number(e.target.value))} className="h-10 rounded-xl border border-[#c9d9ea] bg-white px-3 text-sm"><option disabled>Saved benchmark cases</option>{dashboardData.reports.map((item, index) => <option key={item.reportId} value={index}>{item.reportId} · {item.case}</option>)}</select> : <span className="rounded-xl bg-[#eaf3ff] px-3 py-2 text-sm font-bold text-[#0877ff]">Live case selected</span>}>
    <Panel title="Case header"><div className="grid grid-cols-5 gap-3" style={{ gridTemplateColumns: '2fr 1.2fr 1.3fr 1.1fr 0.7fr' }}><Stat className={centered} label="Case ID" value={String(current.case)} detail={liveMode ? 'Live transaction' : 'Validated artifact'} tone="blue" /><Stat className={centered} label="Account" value={String(current.account || '—')} detail="Selected context" tone="purple" /><Stat className={centered} label="Selected LLM" value="Gemma 4 31B" detail="AML fine-tuned model" tone="green" /><Stat className={centered} label="Decision" value={String(current.decision)} detail={liveReport ? 'Fresh Gemma output' : liveMode ? 'XGBoost escalation' : 'Saved model output'} tone="red" /><Stat className={centered} label={liveMode ? 'XGBoost score' : 'Confidence'} value={liveMode ? liveScore.toFixed(3) : invalidLiveReport ? '—' : `${(Number(current.confidence || 0) * 100).toFixed(0)}%`} detail={liveMode ? 'Escalation threshold 0.300' : invalidLiveReport ? 'Unavailable' : 'Model output'} tone="orange" /></div></Panel>
    <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1.35fr]"><Panel title="Why this case was escalated"><div className="rounded-xl bg-[#fff7e3] p-4 text-lg font-bold text-[#b45309]">{String(current.riskLevel)} · {String(current.pattern)}</div>{current.rationalePoints?.length ? <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-[#48658a]">{current.rationalePoints.map((item: unknown, index: number) => <li key={`${String(item)}-${index}`}>{String(item)}</li>)}</ul> : <p className="mt-4 text-sm leading-6 text-[#48658a]">{String(current.rationale || current.summary)}</p>}<p className="mt-3 text-sm leading-6 text-[#48658a]"><span className="font-bold">Gemma summary:</span> {String(current.summary)}</p>{liveReport && <p className="mt-2 text-sm font-semibold text-[#48658a]"><span className="font-bold">Gemma confidence:</span> {invalidLiveReport ? 'Unavailable for this response' : `${(Number(current.confidence || 0) * 100).toFixed(0)}%`}</p>}<div className="mt-5 flex flex-wrap gap-2"><button onClick={() => go('transactions', context, String(current.account), String(current.case))} className="rounded-xl border border-[#c9d9ea] px-4 py-2 text-sm font-bold text-[#0877ff]">Open transaction evidence</button><button onClick={() => go('graph', context, String(current.account), String(current.case))} className="rounded-xl bg-[#0877ff] px-4 py-2 text-sm font-bold text-white">Open network</button></div></Panel><Panel title="Evidence sources"><ol className="space-y-3">{(current.evidence as unknown[]).map((item, index) => <li key={`${String(item)}-${index}`} className="rounded-xl border border-[#dce7f2] p-3 text-sm text-[#48658a]"><span className="mr-2 inline-grid size-6 place-items-center rounded-full bg-[#eaf3ff] text-xs font-bold text-[#0877ff]">{index + 1}</span>{String(item)}<span className="mt-2 block text-[11px] font-bold uppercase tracking-wide text-[#607590]">Retrieved evidence</span></li>)}</ol></Panel></div>
    <Panel title={liveMode ? 'Generate fresh Gemma report' : 'Re-run saved benchmark case'} className="mt-4"><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-[#48658a]">{liveMode ? 'The selected suspicious transaction will be sent to the Gemma adapter running in Colab.' : 'This is a saved benchmark case. Use it for QA, not as a live prediction.'}</p><button onClick={() => void investigateLive()} disabled={running} className="rounded-xl bg-[#0877ff] px-4 py-2 text-sm font-bold text-white disabled:opacity-50">{running ? 'Running Gemma…' : liveMode ? 'Run fresh Gemma report' : 'Run benchmark case'}</button></div>{liveError && <p className="mt-3 text-sm text-[#b45309]">{liveError}</p>}</Panel>
    <Panel title="Recommended analyst actions" className="mt-4"><div className="grid gap-3 md:grid-cols-3">{(current.actions as unknown[]).map((item, index) => <div key={`${String(item)}-${index}`} className="rounded-xl border border-[#dce7f2] p-4 text-sm text-[#48658a]"><span className="font-bold text-[#10b981]">{index + 1}. </span>{String(item)}</div>)}</div><p className="mt-4 text-xs text-[#607590]">{liveReport ? 'Source: fresh live Gemma inference.' : liveMode ? 'Source: live XGBoost escalation; Gemma report not generated yet.' : 'Source: saved validated DATA 298B Gemma trace.'}</p></Panel>
    <Panel title="Analyst review" className="mt-4"><p className="text-sm text-[#48658a]">Record the analyst disposition separately from the model output. The original XGBoost and Gemma results are preserved.</p><div className="mt-4 flex flex-wrap gap-2"><button onClick={() => setAnalystDecision('CONFIRMED_SUSPICIOUS')} className={`rounded-xl px-4 py-2 text-sm font-bold ${analystDecision === 'CONFIRMED_SUSPICIOUS' ? 'bg-[#f0445e] text-white' : 'border border-[#f7b7c2] text-[#d92d4b]'}`}>Confirm suspicious</button><button onClick={() => setAnalystDecision('MARKED_LEGITIMATE')} className={`rounded-xl px-4 py-2 text-sm font-bold ${analystDecision === 'MARKED_LEGITIMATE' ? 'bg-[#10b981] text-white' : 'border border-[#a7e6cf] text-[#059669]'}`}>Mark legitimate</button><button onClick={() => setAnalystDecision('NEEDS_MORE_EVIDENCE')} className={`rounded-xl px-4 py-2 text-sm font-bold ${analystDecision === 'NEEDS_MORE_EVIDENCE' ? 'bg-[#f59e0b] text-white' : 'border border-[#f4d39a] text-[#b45309]'}`}>Needs more evidence</button></div><textarea value={analystNote} onChange={(e) => setAnalystNote(e.target.value)} className="mt-4 min-h-24 w-full rounded-xl border border-[#c9d9ea] p-3 text-sm text-[#48658a]" placeholder="Analyst note: explain the decision using the transaction and network evidence." /><div className="mt-3 flex items-center gap-3"><button onClick={saveAnalystReview} disabled={!analystDecision} className="rounded-xl bg-[#0877ff] px-4 py-2 text-sm font-bold text-white disabled:opacity-50">Save analyst review</button>{reviewSaved && <span className="text-sm font-semibold text-[#059669]">Review saved for {String(current.case)}</span>}</div></Panel>
  </PageShell>;
}
