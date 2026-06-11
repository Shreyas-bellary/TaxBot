import { ArrowUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { MAX_QUERY_LENGTH } from "../lib/api";

const MAX_ROWS = 10;
const LINE_HEIGHT_PX = 24;

interface InputBarProps {
  pending: boolean;
  onSend: (query: string) => void;
  autoFocus?: boolean;
}

export function InputBar({ pending, onSend, autoFocus = true }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const trimmed = value.trim();
  const canSend = trimmed.length >= 3 && trimmed.length <= MAX_QUERY_LENGTH && !pending;
  const nearLimit = value.length >= MAX_QUERY_LENGTH - 200;

  // Auto-grow the textarea up to MAX_ROWS.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const max = MAX_ROWS * LINE_HEIGHT_PX;
    textarea.style.height = `${Math.min(textarea.scrollHeight, max)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus && !pending) textareaRef.current?.focus();
  }, [autoFocus, pending]);

  const submit = () => {
    if (!canSend) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <div className="border-t border-border bg-bg px-4 pb-4 pt-3">
      <div className="mx-auto w-full max-w-[820px]">
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-surface px-3 py-2 transition-colors focus-within:border-border-strong">
          <textarea
            ref={textareaRef}
            value={value}
            rows={1}
            maxLength={MAX_QUERY_LENGTH + 100}
            placeholder="Ask a U.S. tax question — forms, deductions, credits, filing…"
            aria-label="Ask a tax question"
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            className="max-h-60 min-h-6 flex-1 resize-none bg-transparent py-0.5 text-[15px] leading-6 text-ink placeholder:text-ink-faint focus:outline-none"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-white transition-all hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-35 dark:text-bg"
          >
            <ArrowUp size={16} />
          </button>
        </div>

        <div className="mt-1.5 flex items-center justify-between px-1 text-[11.5px] text-ink-faint">
          <span>
            Answers are grounded in official IRS documents. Verify before filing.
          </span>
          {nearLimit && (
            <span className={value.length > MAX_QUERY_LENGTH ? "text-danger" : ""}>
              {value.length} / {MAX_QUERY_LENGTH}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
