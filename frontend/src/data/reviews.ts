export type AnalystDecision = 'CONFIRMED_SUSPICIOUS' | 'MARKED_LEGITIMATE' | 'NEEDS_MORE_EVIDENCE';

export type AnalystReview = {
  caseId: string;
  decision: AnalystDecision;
  note: string;
  reviewedAt: string;
};

const REVIEW_KEY = 'data298b-analyst-reviews';

export function loadReviews(): Record<string, AnalystReview> {
  try {
    const value = JSON.parse(localStorage.getItem(REVIEW_KEY) || '{}');
    return value && typeof value === 'object' ? value : {};
  } catch {
    return {};
  }
}

export function getReview(caseId: string) {
  return loadReviews()[caseId];
}

export function saveReview(review: AnalystReview) {
  const reviews = loadReviews();
  reviews[review.caseId] = review;
  localStorage.setItem(REVIEW_KEY, JSON.stringify(reviews));
}
