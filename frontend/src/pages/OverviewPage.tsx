import { useEffect, useState } from 'react';
import dashboardData from '../data/dashboard.json';
import detection from '../data/detection_v2.json';
import PageShell, { Panel, Stat } from '../components/PageShell';
import { go } from '../state';
import { getLivePredictions, getLiveStatus, type LivePrediction } from '../data/live';

export default function OverviewPage() {
  const [liveRows, setLiveRows] = useState<LivePrediction[]>([]);
  const [liveOnline, setLiveOnline] = useState(false);
  const [liveError, setLiveError] = useState('');
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const status = await getLiveStatus();
        if (!active) return;
        setLiveOnline(status.status === 'online' && status.data_available && status.model_available);
        const result = await getLivePredictions(8);
        if (active) { setLiveRows(result.items); setLiveError(''); }
      } catch (error) {
        if (active) { setLiveOnline(false); setLiveError(error instanceof Error ? error.message : 'Live API unavailable'); }
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 4000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  const total = dashboardData.summary.totalTransactions;
  const fraud = dashboardData.summary.fraudTransactions;
  const legitimate = total - fraud;
  const formats = dashboardData.transactions.reduce<Record<string, number>>((acc, row) => { const key = String(row['Payment Format']); acc[key] = (acc[key] ?? 0) + (Number(row['Is Laundering']) === 1 ? 1 : 0); return acc; }, {});
  const maxFormat = Math.max(...Object.values(formats), 1);
  const champion = detection.models.find((item) => item.key === detection.champion) ?? detection.models[0];
  const pipeline = [['XGBoost', 'Transaction risk scoring', 'transactions'], ['Graph features', 'Account and network behavior', 'graph'], ['Leiden detection', 'Community context', 'graph'], ['Gemma investigator', 'Evidence-backed case review', 'llm']] as const;
  return <PageShell title="Fraud Intelligence Overview" subtitle="Move from suspicious activity to network context and analyst review."><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><Stat label="Total transactions" value={total.toLocaleString()} detail="Processed dataset" /><Stat label="Total accounts" value={dashboardData.summary.totalAccounts.toLocaleString()} detail="Graph feature export" tone="green" /><Stat label="Fraud transactions" value={fraud.toLocaleString()} detail="Evaluation statistic" tone="red" /><Stat label="Communities" value={dashboardData.summary.communities.toLocaleString()} detail="Leiden assignments" tone="purple" /></div><Panel title="Live prediction stream" className="mt-4"><div className="mb-3 flex items-center justify-between gap-3 text-sm"><span className={liveOnline ? 'font-bold text-[#10b981]' : 'font-bold text-[#b45309]'}>{liveOnline ? '● Connected' : '● Snapshot mode'}</span><span className="text-xs text-[#607590]">Champion XGBoost (leak-free) replay of the test period · refreshes every 4 seconds</span></div>{liveError && <p className="mb-3 text-xs text-[#b45309]">{liveError}</p>}<div className="overflow-x-auto"><table className="w-full min-w-[780px] text-left text-xs"><thead className="bg-[#f4f8fc] text-[#607590]"><tr><th className="px-3 py-2">Time</th><th className="px-3 py-2">From</th><th className="px-3 py-2">To</th><th className="px-3 py-2">Amount</th><th className="px-3 py-2">Risk</th><th className="px-3 py-2">Prediction</th></tr></thead><tbody>{(liveRows.length ? liveRows : dashboardData.transactions.slice(0, 8)).map((row, index) => <tr key={`${String(row.Timestamp)}-${String(row.src_acct)}-${index}`} className="border-b border-[#dce7f2]"><td className="px-3 py-2">{String(row.Timestamp)}</td><td className="px-3 py-2 font-mono">{String(row.src_acct)}</td><td className="px-3 py-2 font-mono">{String(row.dst_acct)}</td><td className="px-3 py-2">${Number(row['Amount Paid']).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td><td className="px-3 py-2 font-bold">{Number((row as LivePrediction).fraud_probability ?? (row as Record<string, unknown>).xgboost_score ?? 0).toFixed(3)}</td><td className={`px-3 py-2 font-bold ${String((row as LivePrediction).prediction ?? 'LEGITIMATE') === 'SUSPICIOUS' ? 'text-[#f0445e]' : 'text-[#10b981]'}`}>{String((row as LivePrediction).prediction ?? 'LEGITIMATE')}</td></tr>)}</tbody></table></div></Panel><div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[#bcd8f5] bg-[#eaf3ff] p-5"><div><h2 className="text-lg font-bold">Start an investigation</h2><p className="mt-1 text-sm text-[#48658a]">Review the transaction queue, then carry the same account into the network and Gemma investigation views.</p></div><button onClick={() => go('transactions')} className="rounded-xl bg-[#0877ff] px-5 py-3 text-sm font-bold text-white">Explore suspicious cases</button></div><Panel title="Detection pipeline" className="mt-4"><div className="grid gap-4 lg:grid-cols-4">{pipeline.map(([name, detail, target], index) => <button key={name} onClick={() => go(target)} className="rounded-xl border border-[#dce7f2] bg-[#f8fbff] p-4 text-left transition hover:-translate-y-0.5 hover:border-[#0877ff] hover:shadow-md"><span className="mb-4 inline-grid size-8 place-items-center rounded-full bg-[#0877ff] text-sm font-bold text-white">0{index + 1}</span><h3 className="font-bold">{name}</h3><p className="mt-2 text-xs text-[#607590]">{detail}</p><span className="mt-4 inline-block text-xs font-bold text-[#0877ff]">Open →</span></button>)}</div></Panel><div className="mt-4 grid gap-4 xl:grid-cols-2"><Panel title="Detection reference · champion XGBoost"><div className="grid gap-3 sm:grid-cols-2"><Stat label="PR-AUC" value={champion.pr_auc.toFixed(3)} detail="Clean test period" /><Stat label="F1" value={champion.f1.toFixed(3)} detail={`Threshold ${champion.threshold.toFixed(3)} chosen on validation`} tone="purple" /><Stat label="Precision" value={`${(champion.precision * 100).toFixed(1)}%`} detail={`${champion.fp.toLocaleString()} false positives`} tone="orange" /><Stat label="Recall" value={`${(champion.recall * 100).toFixed(1)}%`} detail={`${(champion.scheme_recall! * 100).toFixed(1)}% of laundering schemes caught`} tone="green" /></div></Panel><Panel title="Selected Layer 4 model"><p className="text-2xl font-extrabold text-[#7c3aed]">Gemma 4 31B</p><p className="mt-1 text-sm text-[#48658a]">AML fine-tuned model selected from the frozen four-model benchmark.</p></Panel></div></PageShell>;
}
