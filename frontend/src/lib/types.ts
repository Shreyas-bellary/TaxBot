/** Mirrors the FastAPI `CitedParent` response model. */
export interface CitedParent {
  parent_id: string;
  text_content: string;
  metadata: Record<string, unknown>;
}

/** Ephemeral prior turn sent with ask (never stored on the server). */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_at: string;
}

/** Mirrors the FastAPI `AskResponse` response model. */
export interface AskResponse {
  answer: string;
  citations: string[];
  used_parent_ids: string[];
  parents: CitedParent[];
  matched_child_ids: string[];
  rate_limit?: RateLimitInfo | null;
}

export type MessageRole = "user" | "assistant";

export type MessageStatus = "complete" | "error";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  createdAt: number;
  /** Grounding chunks for assistant messages (empty for refusals/errors). */
  sources?: CitedParent[];
  /** Cited IRS source URLs for assistant messages. */
  citations?: string[];
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

/** Helpers for reading typed values out of parent metadata. */
export function metaString(
  metadata: Record<string, unknown>,
  key: string,
): string | null {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function metaNumber(
  metadata: Record<string, unknown>,
  key: string,
): number | null {
  const value = metadata[key];
  return typeof value === "number" ? value : null;
}

/** Build API history from prior complete messages (excludes the current user turn). */
export function toChatHistory(messages: Message[]): ChatTurn[] {
  return messages
    .filter((message) => message.status === "complete")
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));
}
