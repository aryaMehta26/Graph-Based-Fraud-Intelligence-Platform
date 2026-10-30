import { useEffect, useMemo, useState } from 'react';
import dashboardData from '../data/dashboard.json';
import PageShell, { Panel, Stat } from '../components/PageShell';
import { go } from '../state';
import { getLivePredictions, type LivePrediction } from '../data/live';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
type Tx = (typeof dashboardData.transactions)[number];

export default function TransactionsPage() {
  const [query, setQuery] = useState('');
  const [format, setFormat] = useState('All');
  const [evaluationMode, setEvaluationMode] = useState(false);
  const [selected, setSelected] = useState<Tx | LivePrediction | null>(null);
  const [liveRows, setLiveRows] = useState<LivePrediction[]>([]);
  const [liveError, setLiveError] = useState('');

  useEffect(() => {
    getLivePredictions(50)
      .then((result) => setLiveRows(result.items))
      .catch((error) => setLiveError(error instanceof Error ? error.message : 'Live transaction feed unavailable'));
  }, []);

  const sourceRows = liveRows.length ? liveRows : dashboardData.transactions;
  const formats = [...new Set(sourceRows.map((row) => String(row['Payment Format'])))];
  const filtered = useMemo(() => sourceRows.filter((row) => {
    const text = `${row.src_acct} ${row.dst_acct} ${row.Timestamp}`.toLowerCase();
    return (!query || text.includes(query.toLowerCase())) && (format === 'All' || row['Payment Format'] === format);
  }), [format, query, liveRows]);

  const scoreOf = (row: Record<string, unknown>) => Number(row.xgboost_score);
  const liveCase = (row: Tx | LivePrediction) => row as LivePrediction;

  return <PageShell title="Suspicious Transactions" subtitle="Review live scored activity and send an escalated case to the Gemma investigator.">
    <div className="flex flex-wrap gap-3 rounded-2xl border border-[#dce7f2] bg-white p-4">
      <label className="flex min-w-[280px] flex-1 flex-col gap-1 text-xs font-semibold text-[#607590]">Search<input value={query} onChange={(e) => setQuery(e.target.value)} className="mt-1 h-10 rounded-xl border border-[#c9d9ea] px-3 text-sm font-normal outline-none focus:border-[#0877ff]" placeholder="Account ID or timestamp" /></label>
      <label className="flex w-44 flex-col gap-1 text-xs font-semibold text-[#607590]">Payment format<select value={format} onChange={(e) => setFormat(e.target.value)} className="mt-1 h-10 rounded-xl border border-[#c9d9ea] bg-white px-2 text-sm font-normal"><option>All</option>{formats.map((item) => <option key={item}>{item}</option>)}</select></label>
      <label className="flex items-end gap-2 pb-2 text-xs text-[#607590]"><input type="checkbox" checked={evaluationMode} onChange={(e) => setEvaluationMode(e.target.checked)} /> Evaluation mode</label>
    </div>
    <div className="mt-4 grid gap-4 sm:grid-cols-3"><Stat label="Rows in review" value={String(filtered.length)} detail={liveRows.length ? 'Live replay window' : 'Saved preview'} /><Stat label="Analyst view" value="Suspicious queue" detail="Live Gemma is on-demand" tone="green" /><Stat label="Source" value={liveRows.length ? 'Live API replay' : 'Snapshot fallback'} detail={liveError || 'XGBoost scored transactions'} tone="purple" /></div>
    <Panel title="Transaction review queue" className="mt-4"><p className="mb-4 text-xs text-[#607590]">The live API scores transactions with XGBoost. Select a suspicious row to pass that exact transaction into a fresh Gemma investigation.</p><div className="overflow-x-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="bg-[#f4f8fc] text-[#607590]"><tr>{['Timestamp', 'From account', 'To account', 'Amount', 'Format', 'Risk signal', 'Action'].map((head) => <th key={head} className="px-3 py-3">{head}</th>)}{evaluationMode && <th className="px-3 py-3">Ground truth</th>}</tr></thead><tbody>{filtered.map((row, index) => { const score = scoreOf(row as Record<string, unknown>); const live = liveCase(row); const suspicious = Number.isFinite(score) && score >= 0.3; return <tr key={`${row.Timestamp}-${row.src_acct}-${index}`} className="cursor-pointer border-b border-[#dce7f2] hover:bg-[#f8fbff]" onClick={() => setSelected(row)}><td className="px-3 py-3">{String(row.Timestamp)}</td><td className="px-3 py-3 font-mono">{row.src_acct}</td><td className="px-3 py-3 font-mono">{row.dst_acct}</td><td className="px-3 py-3">{money.format(Number(row['Amount Paid']))}</td><td className="px-3 py-3">{row['Payment Format']}</td><td className="px-3 py-3"><span className={`font-bold ${suspicious ? 'text-[#f0445e]' : 'text-[#10b981]'}`}>{Number.isFinite(score) ? `${score.toFixed(3)} · ${suspicious ? 'SUSPICIOUS' : 'LEGITIMATE'}` : 'Not in snapshot'}</span></td><td className="px-3 py-3"><button onClick={(e) => { e.stopPropagation(); setSelected(row); }} className="font-bold text-[#0877ff]">{live.investigatable || suspicious ? 'Investigate' : 'View details'}</button></td>{evaluationMode && <td className="px-3 py-3">{Number((row as Record<string, unknown>)['Is Laundering']) === 1 ? 'SUSPICIOUS' : 'LEGITIMATE'}</td>}</tr>; })}</tbody></table></div></Panel>
    {selected && <div className="fixed inset-0 z-30 bg-[#071a30]/30 p-4" onClick={() => setSelected(null)}><div className="ml-auto h-full max-w-xl overflow-auto rounded-2xl bg-white p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}><div className="flex items-start justify-between"><div><p className="text-xs font-bold uppercase tracking-wide text-[#607590]">Transaction detail</p><h2 className="mt-1 text-xl font-extrabold">{selected.src_acct} → {selected.dst_acct}</h2></div><button onClick={() => setSelected(null)} className="text-2xl text-[#607590]">×</button></div><dl className="mt-6 grid grid-cols-2 gap-4 text-sm"><Detail label="Timestamp" value={String(selected.Timestamp)} /><Detail label="Amount" value={money.format(Number(selected['Amount Paid']))} /><Detail label="Payment format" value={String(selected['Payment Format'])} /><Detail label="Source" value={liveRows.length ? 'Live API replay' : 'Saved pipeline snapshot'} /></dl><div className="mt-6 flex flex-wrap gap-2"><button onClick={() => go('graph', selected, String(selected.src_acct))} className="rounded-xl bg-[#0877ff] px-4 py-2 text-sm font-bold text-white">View sender network</button><button onClick={() => go('graph', selected, String(selected.dst_acct))} className="rounded-xl border border-[#c9d9ea] px-4 py-2 text-sm font-bold text-[#0877ff]">View receiver network</button>{scoreOf(selected as Record<string, unknown>) >= 0.3 && <button onClick={() => go('llm', selected, String(selected.src_acct), String((selected as Record<string, unknown>).case_id || `live-${selected.Timestamp}`))} className="rounded-xl border border-[#c9d9ea] px-4 py-2 text-sm font-bold text-[#0877ff]">Investigate live case</button>}</div></div></div>}
  </PageShell>;
}

function Detail({ label, value }: { label: string; value: string }) { return <div><dt className="text-xs text-[#607590]">{label}</dt><dd className="mt-1 break-all font-semibold">{value}</dd></div>; }
