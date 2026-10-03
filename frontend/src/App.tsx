import { useEffect, useState } from 'react';
import OverviewPage from './pages/OverviewPage';
import TransactionsPage from './pages/TransactionsPage';
import GraphPage from './pages/GraphPage';
import LLMPage from './pages/LLMPage';
import ModelCompPage from './pages/ModelCompPage';
import ReportsPage from './pages/ReportsPage';

type Page = 'overview' | 'transactions' | 'graph' | 'llm' | 'reports' | 'modelcomp';
const pages: { id: Page; label: string; icon: string }[] = [{ id: 'overview', label: 'Overview', icon: '▦' }, { id: 'transactions', label: 'Transactions', icon: '⇄' }, { id: 'graph', label: 'Network', icon: '⌘' }, { id: 'llm', label: 'LLM Investigator', icon: '✦' }, { id: 'modelcomp', label: 'Model Comparison', icon: '▥' }, { id: 'reports', label: 'Reports', icon: '▤' }];
const pageFromHash = (): Page => { const value = window.location.hash.replace(/^#\//, ''); return pages.some((item) => item.id === value) ? value as Page : 'overview'; };

function Sidebar({ current, onNavigate }: { current: Page; onNavigate: (page: Page) => void }) {
  return <aside className="flex h-full w-[220px] shrink-0 flex-col bg-[#071a30] px-4 py-5 text-white"><button className="mb-8 flex items-center gap-3 rounded-xl px-2 text-left" onClick={() => onNavigate('overview')}><span className="grid size-10 place-items-center rounded-xl bg-[#0877ff] text-lg font-extrabold">FI</span><span><strong className="block text-sm">Fraud Intelligence</strong><small className="text-[10px] text-[#9eb2c9]">AML Investigation Platform</small></span></button><nav className="space-y-2" aria-label="Main navigation">{pages.map((item) => <button key={item.id} onClick={() => onNavigate(item.id)} aria-current={current === item.id ? 'page' : undefined} className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${current === item.id ? 'bg-[#0877ff] text-white shadow-lg' : 'text-[#b5c5d8] hover:bg-[#102b49] hover:text-white'}`}><span className="w-5 text-center text-lg">{item.icon}</span>{item.label}</button>)}</nav><div className="mt-auto rounded-xl border border-[#26425f] p-3 text-[11px] text-[#9eb2c9]">Source<br /><span className="font-semibold text-white">Existing pipeline artifacts</span><br /><span>Refresh with the local launcher</span></div></aside>;
}

export default function App() {
  const [current, setCurrent] = useState<Page>(pageFromHash);
  useEffect(() => { const onChange = () => setCurrent(pageFromHash()); window.addEventListener('hashchange', onChange); return () => window.removeEventListener('hashchange', onChange); }, []);
  const navigate = (page: Page) => { window.location.hash = `#/${page}`; setCurrent(page); };
  const content: Record<Page, React.ReactElement> = { overview: <OverviewPage />, transactions: <TransactionsPage />, graph: <GraphPage />, llm: <LLMPage />, modelcomp: <ModelCompPage />, reports: <ReportsPage /> };
  return <div className="flex h-screen min-h-0 w-screen overflow-hidden"><Sidebar current={current} onNavigate={navigate} /><main id="dashboard-scroll" className="min-w-0 min-h-0 flex-1 overflow-y-auto overflow-x-hidden">{content[current]}</main></div>;
}
