import type { AskResponse, ChatTurn, RateLimitInfo } from "./types";

const configuredApiBase = (
  import.meta.env.VITE_API_BASE_URL as string | undefined
)?.replace(/\/+$/, "");
const API_BASE: string =
  configuredApiBase ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

export const MAX_QUERY_LENGTH = 2000;
/** Max prior turns sent with each ask (must match backend MAX_HISTORY_TURNS). */
export const MAX_HISTORY_TURNS = 12;

/** A user-presentable API failure; `message` reads like an assistant reply. */
export class AskError extends Error {
  readonly status: number | null;
  readonly rateLimit: RateLimitInfo | null;

  constructor(
    message: string,
    status: number | null,
    rateLimit: RateLimitInfo | null = null,
  ) {
    super(message);
    this.name = "AskError";
    this.status = status;
    this.rateLimit = rateLimit;
  }
}

function friendlyMessage(status: number, detail: string): string {
  switch (status) {
    case 400:
      return "I couldn't process that question — it looks like it contains unsupported instructions. Please rephrase it as a plain tax question.";
    case 403:
      return "That request was blocked by a safety check. Please rephrase your question and try again.";
    case 404:
      return "I couldn't find any matching IRS content for that question. Try mentioning a specific form, publication, or tax topic.";
    case 429:
      return (
        detail ||
        "You've used today's free answers. Please try again tomorrow (quota resets at midnight UTC)."
      );
    case 502:
      return "I generated an answer but couldn't verify its sources, so I'm not showing it. Please try asking again.";
    default:
      return detail || "Something went wrong while answering. Please try again.";
  }
}

function parseRateLimitHeaders(response: Response): RateLimitInfo | null {
  const limit = response.headers.get("X-RateLimit-Limit");
  const remaining = response.headers.get("X-RateLimit-Remaining");
  const resetAt = response.headers.get("X-RateLimit-Reset");
  if (limit === null || remaining === null) return null;
  const limitN = Number(limit);
  const remainingN = Number(remaining);
  if (!Number.isFinite(limitN) || !Number.isFinite(remainingN)) return null;
  return {
    limit: limitN,
    remaining: remainingN,
    reset_at: resetAt ?? "",
  };
}

function parseDetail(body: unknown): string {
  if (
    typeof body === "object" &&
    body !== null &&
    typeof (body as { detail?: unknown }).detail === "string"
  ) {
    return (body as { detail: string }).detail;
  }
  return "";
}


export async function ask(
  query: string,
  history: ChatTurn[] = [],
): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history: history.slice(-MAX_HISTORY_TURNS),
      }),
    });
  } catch {
    throw new AskError(
      "I couldn't reach the TaxBot service at the moment. Please try again later.",
      null,
    );
  }

  const headerQuota = parseRateLimitHeaders(response);

  if (!response.ok) {
    let detail = "";
    try {
      detail = parseDetail(await response.json());
    } catch {
      // Non-JSON error body; fall through to the generic message.
    }
    throw new AskError(
      friendlyMessage(response.status, detail),
      response.status,
      headerQuota,
    );
  }

  const body = (await response.json()) as AskResponse;
  if (!body.rate_limit && headerQuota) {
    body.rate_limit = headerQuota;
  }
  return body;
}

/** GET /v1/rate-limit — remaining free answers for this IP today. */
export async function fetchRateLimit(): Promise<RateLimitInfo | null> {
  try {
    const response = await fetch(`${API_BASE}/v1/rate-limit`);
    if (!response.ok) return null;
    const body = (await response.json()) as {
      enabled?: boolean;
      limit?: number;
      remaining?: number;
      reset_at?: string;
    };
    if (body.enabled === false) return null;
    if (
      typeof body.limit !== "number" ||
      typeof body.remaining !== "number"
    ) {
      return null;
    }
    return {
      limit: body.limit,
      remaining: body.remaining,
      reset_at: body.reset_at ?? "",
    };
  } catch {
    return null;
  }
}
