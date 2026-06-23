import {
  Children,
  Fragment,
  cloneElement,
  isValidElement,
  useMemo,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { dedupeCitationsToLastOccurrence, collapseAdjacentCitationRuns, normalizeCitationPlacement, sourceForDocIndex, splitDocCitations } from "../lib/citations";
import type { CitedParent } from "../lib/types";
import { CitationChip } from "./CitationChip";

interface CitationMarkdownProps {
  content: string;
  sources: CitedParent[];
  onOpenSource: (anchorId: string) => void;
}

type MarkdownTag =
  | "p"
  | "li"
  | "td"
  | "th"
  | "strong"
  | "em"
  | "h1"
  | "h2"
  | "h3"
  | "h4"
  | "blockquote";

const CITATION_WRAPPER_TAGS: MarkdownTag[] = [
  "p",
  "li",
  "td",
  "th",
  "strong",
  "em",
  "h1",
  "h2",
  "h3",
  "h4",
  "blockquote",
];

function injectCitationChips(
  text: string,
  sources: CitedParent[],
  onOpenSource: (anchorId: string) => void,
): ReactNode[] {
  const parts = splitDocCitations(text);
  const nodes: ReactNode[] = [];
  let skipNext = false;

  for (let index = 0; index < parts.length; index += 1) {
    if (skipNext) {
      skipNext = false;
      continue;
    }

    const part = parts[index];

    if (typeof part === "string") {
      if (part.length > 0) nodes.push(<Fragment key={index}>{part}</Fragment>);
      continue;
    }

    const source = sourceForDocIndex(sources, part);
    const next = parts[index + 1];
    const trailingMatch =
      typeof next === "string" ? next.match(/^[:.,;!?]+/) : null;

    if (!source) {
      nodes.push(<Fragment key={index}>[Doc-{part}]</Fragment>);
      continue;
    }

    const chip = (
      <CitationChip
        source={source}
        onOpen={() => onOpenSource(source.parent_id)}
      />
    );

    if (trailingMatch && typeof next === "string") {
      const trailingPunct = trailingMatch[0];
      const remainder = next.slice(trailingPunct.length);
      nodes.push(
        <span key={index} className="inline-flex items-baseline whitespace-nowrap">
          {chip}
          <span className="text-inherit">{trailingPunct}</span>
        </span>,
      );
      if (remainder.length > 0) {
        nodes.push(<Fragment key={`${index}-rest`}>{remainder}</Fragment>);
      }
      skipNext = true;
      continue;
    }

    nodes.push(<Fragment key={index}>{chip}</Fragment>);
  }

  return nodes;
}

function processChildren(
  children: ReactNode,
  sources: CitedParent[],
  onOpenSource: (anchorId: string) => void,
): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === "string") {
      return injectCitationChips(child, sources, onOpenSource);
    }

    if (isValidElement<{ children?: ReactNode }>(child) && child.props.children != null) {
      return cloneElement(child, {
        ...child.props,
        children: processChildren(child.props.children, sources, onOpenSource),
      });
    }

    return child;
  });
}

function wrapTag(tag: MarkdownTag) {
  const Tag = tag;
  return function CitationWrapper({
    children,
    ...props
  }: React.HTMLAttributes<HTMLElement> & { children?: ReactNode }) {
    return <Tag {...props}>{children}</Tag>;
  };
}

export function CitationMarkdown({
  content,
  sources,
  onOpenSource,
}: CitationMarkdownProps) {
  const displayContent = useMemo(
    () =>
      collapseAdjacentCitationRuns(
        normalizeCitationPlacement(dedupeCitationsToLastOccurrence(content)),
      ),
    [content],
  );

  const components = useMemo(() => {
    const wrapped: Record<string, ReturnType<typeof wrapTag>> = {};

    for (const tag of CITATION_WRAPPER_TAGS) {
      const Base = wrapTag(tag);
      wrapped[tag] = function CitationComponent({
        children,
        ...props
      }: React.HTMLAttributes<HTMLElement> & { children?: ReactNode }) {
        return (
          <Base {...props}>
            {processChildren(children, sources, onOpenSource)}
          </Base>
        );
      };
    }

    return wrapped;
  }, [sources, onOpenSource]);

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {displayContent}
    </ReactMarkdown>
  );
}
