import type { ReactNode } from 'react';

export default function PageShell({ title, subtitle, children, action }: { title: string; subtitle: string; children: ReactNode; action?: ReactNode }) {
  return <div className="min-h-full bg-[#f5f9fd] text-[#10233d]"><header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-[#dce7f2] bg-white/95 px-6 backdrop-blur"><div><span className="text-xs font-bold text-[#607590]">Fraud Intelligence Platform</span><span className="ml-3 rounded-full bg-[#eaf3ff] px-2 py-1 text-[10px] text-[#0877ff]">DATA 298B</span></div><div className="text-xs text-[#607590]">Team 2　·　SJSU　·　HI-Medium</div></header><div className="mx-auto w-full max-w-[1600px] px-6 py-6"><div className="mb-5 flex flex-wrap items-end justify-between gap-4"><div><h1 className="text-3xl font-extrabold tracking-tight text-[#10233d]">{title}</h1><p className="mt-1 text-sm text-[#607590]">{subtitle}</p></div>{action}</div>{children}</div></div>;
}

export function Panel({ title, children, className = '' }: { title?: string; children: ReactNode; className?: string }) {
  return <section className={`rounded-2xl border border-[#dce7f2] bg-white p-5 shadow-[0_5px_18px_rgba(23,59,101,0.06)] ${className}`}>{title && <h2 className="mb-4 text-lg font-bold text-[#10233d]">{title}</h2>}{children}</section>;
}

export function Stat({ label, value, detail, tone = 'blue', className = '' }: { label: string; value: string; detail?: string; tone?: 'blue' | 'red' | 'green' | 'purple' | 'orange'; className?: string }) {
  const styles = { blue: 'bg-[#eaf3ff] text-[#0877ff]', red: 'bg-[#fff0f2] text-[#f0445e]', green: 'bg-[#e9fbf4] text-[#10b981]', purple: 'bg-[#f2eafe] text-[#7c3aed]', orange: 'bg-[#fff7e3] text-[#ff8a34]' };
  return <div className={`min-w-0 rounded-2xl border border-[#dce7f2] bg-white p-4 shadow-sm ${className}`}><div className={`mb-3 inline-flex max-w-full rounded-xl px-3 py-2 text-xs font-bold ${styles[tone]}`}>{label}</div><div className="min-w-0 break-all text-2xl font-extrabold leading-tight text-[#10233d]" title={value}>{value}</div>{detail && <div className="mt-1 text-xs text-[#607590]">{detail}</div>}</div>;
}
