import { useMemo, useState } from 'react';
import agenticReports from '../data/agentic_reports.json';
import PageShell, { Panel, Stat } from '../components/PageShell';
import { postApi } from '../data/api';

type Report = (typeof agenticReports)[number];
const models = ['Gemma', 'Qwen', 'Ministral', 'Granite', 'Phi', 'Llama 3.1 8B'];
const asList = (value: unknown): string[] => Array.isArray(value) ? value.map(String) : value ? [String(value)] : [];

export default function LLMPage() {
  const [model, setModel] = useState(models[0]);
  const [caseIndex, setCaseIndex] = useState(0);
  const [loaded, setLoaded] = useState(true);
  const [provider, setProvider] = useState('saved');
  const [liveReport, setLiveReport] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [liveCase] = useState<any>(() => {
    try { return JSON.parse(sessionStorage.getItem('live_case_context') ?? 'null'); } catch { return null; }
  });
  const [useLiveCase, setUseLiveCase] = useState(Boolean(liveCase));
  const modelReports = useMemo(() => {
    const matches = agenticReports.filter((item) => item.model === model);
    return matches.length ? matches : agenticReports.filter((item) => item.model === 'Gemma');
  }, [model]);
  const savedReport: Report | undefined = modelReports[caseIndex] ?? modelReports[0];
  const report = useLiveCase && liveCase && savedReport ? {
    ...savedReport,
    caseId: liveCase.case_id,
    pattern: liveCase.case_pattern ?? 'CONNECTED_ACTIVITY',
    riskLevel: String(liveCase.risk_level ?? 'MEDIUM').toUpperCase(),
    summary: `Live case containing ${liveCase.case_transaction_ids?.length ?? 1} connected pipeline transactions across ${liveCase.case_account_ids?.length ?? 0} accounts.`,
    evidence: [],
    actions: [],
  } : savedReport;
  const chooseModel = (value: string) => { setModel(value); setCaseIndex(0); setLoaded(false); setLiveReport(null); setError(''); };
  const runInvestigation = async () => {
    if (!report) return;
    setRunning(true); setError('');
    try {
      const context = useLiveCase && liveCase ? { case_id: liveCase.case_id, pattern: liveCase.case_pattern, risk_level: liveCase.risk_level, transactions: liveCase.case_transactions, transaction_ids: liveCase.case_transaction_ids, account_ids: liveCase.case_account_ids, evidence: [], recommended_actions: [], summary: `Live case containing ${liveCase.case_transaction_ids?.length ?? 1} connected pipeline transactions.` } : { case_id: report.caseId, pattern: report.pattern, risk_level: report.riskLevel, evidence: report.evidence, recommended_actions: report.actions, summary: report.summary };
      const result = await postApi<any>('/api/investigations', { provider, model, case_id: context.case_id, context });
      setLiveReport(result); setLoaded(true);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Investigation failed'); }
    finally { setRunning(false); }
  };
  if (!report) return <PageShell title="AI Investigation Assistant" subtitle="No normalized LLM reports are available." />;
  const liveTransactionEvidence = liveCase?.case_transactions?.map((item: any) => `${item.event_id}: ${item.src_acct} → ${item.dst_acct}, $${Number(item.amount ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}, risk ${Number(item.risk_score ?? 0).toFixed(3)}, ${item.prediction ?? 'Unknown'}.`) ?? [];
  const displayed = liveReport ? { ...report, caseId: liveReport.case_id ?? liveCase?.case_id ?? report.caseId, model: liveReport.model ?? report.model, riskLevel: String(liveReport.risk_level ?? report.riskLevel).toUpperCase(), pattern: liveReport.pattern ?? report.pattern, confidence: Number(liveReport.confidence ?? report.confidence), summary: liveReport.summary ?? report.summary, evidence: asList(liveReport.evidence).length ? asList(liveReport.evidence) : liveTransactionEvidence, actions: asList(liveReport.recommended_actions ?? liveReport.actions ?? report.actions), decision: liveReport.decision ?? report.decision } : { ...report, evidence: useLiveCase && liveCase ? liveTransactionEvidence : asList(report.evidence), actions: asList(report.actions) };
  return <PageShell title="AI Investigation Assistant" subtitle="Run the same case through saved traces, fine-tuned adapters, Ollama, or Claude using one investigator contract." action={<div className="flex items-center gap-2 text-xs font-semibold text-[#607590]"><label>Model<select value={model} onChange={(event) => chooseModel(event.target.value)} className="ml-2 h-10 rounded-xl border border-[#c9d9ea] bg-white px-3 text-sm text-[#071d3d]">{models.map((item) => <option key={item}>{item}</option>)}</select></label><label>Provider<select value={provider} onChange={(event) => { setProvider(event.target.value); setLiveReport(null); }} className="ml-2 h-10 rounded-xl border border-[#c9d9ea] bg-white px-3 text-sm text-[#071d3d]"><option value="saved">Saved artifact</option><option value="transformers">Fine-tuned adapter</option><option value="ollama">Ollama</option><option value="claude">Claude</option></select></label></div>}>
    <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
      <Panel title="Investigation input"><p className="mb-4 text-xs text-[#607590]">Choose a case and run it through the selected provider.</p>{liveCase && <div className="mb-4 rounded-xl border border-[#b9d8ff] bg-[#eaf3ff] p-3 text-xs"><strong className="text-[#0877ff]">Live case loaded: {liveCase.case_id}</strong><p className="mt-1 text-[#607590]">{liveCase.case_transaction_ids?.length ?? 1} related transactions · {liveCase.case_account_ids?.length ?? 0} accounts · {liveCase.case_pattern}</p></div>}<label className="text-xs font-semibold text-[#607590]">Case<select value={useLiveCase && liveCase ? 'live' : String(caseIndex)} onChange={(event) => { if (event.target.value === 'live') { setUseLiveCase(true); } else { setUseLiveCase(false); setCaseIndex(Number(event.target.value)); } setLoaded(false); setLiveReport(null); }} className="mt-2 h-11 w-full rounded-xl border border-[#c9d9ea] bg-white px-3 text-sm">{liveCase && <option value="live">{liveCase.case_id} · {liveCase.case_pattern} · Live</option>}{modelReports.map((item, index) => <option key={item.id} value={index}>{item.caseId} · {item.pattern}</option>)}</select></label><button onClick={runInvestigation} disabled={running} className="mt-5 w-full rounded-xl bg-[#0877ff] px-4 py-3 text-sm font-bold text-white hover:bg-[#0666db] disabled:opacity-60">{running ? 'Running…' : provider === 'saved' ? 'Load investigation' : `Run ${provider}`}</button>{error && <p className="mt-3 rounded-lg bg-[#fff0f2] p-3 text-xs text-[#f0445e]">{error}</p>}<div className="mt-5 grid grid-cols-2 gap-3"><Stat label="Model" value={displayed.model} tone="purple" /><Stat label="Latency" value={`${Number(liveReport?.latency_seconds ?? displayed.latencySeconds).toFixed(1)}s`} tone="blue" /><Stat label="Confidence" value={`${(Number(displayed.confidence) * 100).toFixed(0)}%`} tone="green" /><Stat label="Evidence" value={String(displayed.evidence.length)} tone="orange" /></div></Panel>
      <Panel title="Investigation result"><div className={`mb-5 rounded-xl px-4 py-3 text-sm font-bold ${displayed.riskLevel === 'HIGH' || displayed.riskLevel === 'CRITICAL' ? 'bg-[#fff0f2] text-[#f0445e]' : 'bg-[#eaf4ff] text-[#0877ff]'}`}>{displayed.riskLevel} · {displayed.pattern} · {displayed.model}</div><div className="grid gap-5 lg:grid-cols-2"><div><h3 className="mb-3 font-bold">Summary</h3><p className="text-sm leading-6 text-[#48658a]">{displayed.summary || 'No summary was returned for this case.'}</p><dl className="mt-5 space-y-2 text-sm"><div className="flex justify-between gap-4"><dt className="text-[#607590]">Decision</dt><dd className="font-bold">{displayed.decision}</dd></div><div className="flex justify-between gap-4"><dt className="text-[#607590]">Pattern</dt><dd className="font-bold">{displayed.pattern}</dd></div><div className="flex justify-between gap-4"><dt className="text-[#607590]">Case</dt><dd className="font-mono">{displayed.caseId}</dd></div></dl></div><div><h3 className="mb-3 font-bold">Evidence</h3><ol className="space-y-3">{displayed.evidence.map((item: string, index: number) => <li key={`${item}-${index}`} className="flex gap-3 text-sm text-[#48658a]"><span className="grid size-6 shrink-0 place-items-center rounded-full bg-[#eaf3ff] text-xs font-bold text-[#0877ff]">{index + 1}</span>{item}</li>)}</ol><h3 className="mb-3 mt-6 font-bold">Recommended actions</h3><ul className="space-y-3">{displayed.actions.map((item: string, index: number) => <li key={`${item}-${index}`} className="rounded-xl border border-[#dce7f2] p-3 text-sm text-[#48658a]"><span className="mr-2 font-bold text-[#10b981]">{index + 1}.</span>{item}</li>)}</ul></div></div></Panel>
    </div>
  </PageShell>;
}
