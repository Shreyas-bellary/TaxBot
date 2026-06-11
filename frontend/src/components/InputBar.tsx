import { ArrowUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { MAX_QUERY_LENGTH } from "../lib/api";

const MAX_ROWS = 10;
const LINE_HEIGHT_PX = 24;
const SHELL_PADDING_X = "px-4";

export const SAMPLE_QUESTIONS = [
  "What is the standard deduction for tax year 2024?",
  "How do I file Form 1040 if I'm self-employed?",
  "Who qualifies for the Earned Income Tax Credit?",
  "What are the estimated tax payment deadlines?",
];

interface InputBarProps {
  pending: boolean;
  showSuggestions?: boolean;
  docked?: boolean;
  onSend: (query: string) => void;
  onAskSample?: (query: string) => void;
  autoFocus?: boolean;
}

export function InputBar({
  pending,
  showSuggestions = false,
  docked = true,
  onSend,
  onAskSample,
  autoFocus = true,
}: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const trimmed = value.trim();
  const canSend = trimmed.length >= 3 && trimmed.length <= MAX_QUERY_LENGTH && !pending;
  const nearLimit = value.length >= MAX_QUERY_LENGTH - 200;

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
    <div className="relative mx-auto w-full max-w-[820px]">
      {docked && (
        <div className="pointer-events-none absolute inset-x-0 -top-12 h-12 bg-gradient-to-t from-bg to-transparent" />
      )}

      <div
        className={`input-shell flex items-end gap-2 rounded-[26px] bg-surface py-3 shadow-[0_1px_2px_rgba(21,34,56,0.05)] dark:shadow-[0_1px_3px_rgba(0,0,0,0.5)] ${SHELL_PADDING_X}`}
      >
        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          maxLength={MAX_QUERY_LENGTH + 100}
          placeholder="Ask any U.S. tax question"
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
          className={`submit-btn flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-white transition-all duration-300 ease-out hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-30 dark:text-bg ${
            canSend ? "scale-100 hover:scale-105 active:scale-90" : "scale-95 opacity-30"
          }`}
        >
          <ArrowUp size={16} />
        </button>
      </div>

      <div
        className={`grid transition-all duration-400 ease-out ${
          showSuggestions
            ? "mt-4 grid-rows-[1fr] opacity-100"
            : "mt-0 grid-rows-[0fr] opacity-0"
        }`}
        aria-hidden={!showSuggestions}
      >
        <div className="overflow-hidden">
          <div
            className={`flex flex-col items-start gap-0.5 ${SHELL_PADDING_X}`}
            role="group"
            aria-label="Suggested questions"
          >
            {SAMPLE_QUESTIONS.map((question, index) => (
              <button
                key={question}
                type="button"
                disabled={pending || !showSuggestions}
                onClick={() => onAskSample?.(question)}
                style={{ animationDelay: `${index * 50}ms` }}
                className="suggestion-link m-0 w-fit max-w-full border-0 bg-transparent py-1.5 text-left text-[13.5px] leading-snug text-ink-faint transition-colors duration-200 hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>

      {docked && (
        <p className="disclaimer-fade mt-2 text-center text-[11.5px] text-ink-faint">
          {nearLimit ? (
            <span className={value.length > MAX_QUERY_LENGTH ? "text-danger" : ""}>
              {value.length} / {MAX_QUERY_LENGTH} — Answers are grounded in official
              IRS documents. Verify before filing.
            </span>
          ) : (
            "Answers are grounded in official IRS documents. Verify before filing."
          )}
        </p>
      )}
    </div>
  );
}
