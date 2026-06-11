import type { AskResponse } from "./types";

const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

export const MAX_QUERY_LENGTH = 2000;

/** A user-presentable API failure; `message` reads like an assistant reply. */
export class AskError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "AskError";
    this.status = status;
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
    case 502:
      return "I generated an answer but couldn't verify its sources, so I'm not showing it. Please try asking again.";
    default:
      return detail || "Something went wrong while answering. Please try again.";
  }
}

/** POST /v1/ask. Throws `AskError` with a friendly message on failure. */
export async function ask(query: string): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch {
    throw new AskError(
      "I couldn't reach the TaxBot service at the moment. Please try again later.",
      null,
    );
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        typeof (body as { detail?: unknown }).detail === "string"
      ) {
        detail = (body as { detail: string }).detail;
      }
    } catch {
      // Non-JSON error body; fall through to the generic message.
    }
    throw new AskError(friendlyMessage(response.status, detail), response.status);
  }

  return (await response.json()) as AskResponse;
}
