export type SelectedTransaction = Record<string, unknown> & {
  Timestamp?: string;
  src_acct?: string;
  dst_acct?: string;
  txn_id?: string;
};

const TRANSACTION_KEY = 'aml:selected-transaction';
const ACCOUNT_KEY = 'aml:selected-account';
const CASE_KEY = 'aml:selected-case';

export function saveContext(transaction?: SelectedTransaction | null, account?: string | null, caseId?: string | null) {
  if (transaction) localStorage.setItem(TRANSACTION_KEY, JSON.stringify(transaction));
  if (account) localStorage.setItem(ACCOUNT_KEY, account);
  if (caseId) localStorage.setItem(CASE_KEY, caseId);
}

export function loadTransaction(): SelectedTransaction | null {
  try { return JSON.parse(localStorage.getItem(TRANSACTION_KEY) || 'null') as SelectedTransaction | null; } catch { return null; }
}

export function loadAccount(): string {
  return localStorage.getItem(ACCOUNT_KEY) || '';
}

export function loadCaseId(): string {
  return localStorage.getItem(CASE_KEY) || '';
}

export function go(page: string, transaction?: SelectedTransaction | null, account?: string | null, caseId?: string | null) {
  saveContext(transaction, account, caseId);
  window.location.hash = `#/${page}`;
}
