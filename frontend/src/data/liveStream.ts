import { useEffect, useState } from 'react';

export type LivePrediction = {
  event_id: string;
  timestamp: string;
  received_at?: string;
  case_id?: string;
  case_pattern?: string;
  case_transaction_ids?: string[];
  case_account_ids?: string[];
  case_transactions?: LivePrediction[];
  src_acct: string;
  dst_acct: string;
  amount: number;
  payment_format: string;
  risk_score: number;
  prediction: 'Suspicious' | 'Legitimate';
  risk_level: 'High' | 'Medium' | 'Low';
  source: string;
};

export type LiveStreamStatus = 'connecting' | 'live' | 'reconnecting' | 'offline';

const baseUrl = (import.meta.env.VITE_STREAM_API_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://127.0.0.1:8765';

export function useLivePredictions(enabled = true) {
  const [events, setEvents] = useState<LivePrediction[]>([]);
  const [status, setStatus] = useState<LiveStreamStatus>(enabled ? 'connecting' : 'offline');

  useEffect(() => {
    if (!enabled) return undefined;
    const stream = new EventSource(`${baseUrl}/api/stream?interval=2`);
    stream.onopen = () => setStatus('live');
    stream.onmessage = (message) => {
      try {
        const event = { ...JSON.parse(message.data), received_at: new Date().toISOString() } as LivePrediction;
        setEvents((current) => [event, ...current].slice(0, 12));
      } catch {
        // Ignore malformed demo events without interrupting the stream.
      }
    };
    stream.onerror = () => setStatus('reconnecting');
    return () => stream.close();
  }, [enabled]);

  return { events, status };
}
