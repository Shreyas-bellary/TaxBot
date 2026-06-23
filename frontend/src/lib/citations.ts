import { metaNumber, type CitedParent } from "./types";

export const DOC_CITATION_RE = /\[Doc-(\d+)\]/g;

export function sourceForDocIndex(
  sources: CitedParent[],
  docIndex: number,
): CitedParent | undefined {
  const byDocIndex = sources.find(
    (source) => metaNumber(source.metadata, "doc_index") === docIndex,
  );
  if (byDocIndex) return byDocIndex;

  if (docIndex >= 1 && docIndex <= sources.length) {
    return sources[docIndex - 1];
  }
  return undefined;
}

export function citedDocIndices(content: string): number[] {
  const indices = new Set<number>();
  const re = new RegExp(DOC_CITATION_RE.source, "g");
  let match: RegExpExecArray | null;

  while ((match = re.exec(content)) !== null) {
    indices.add(Number(match[1]));
  }

  return [...indices].sort((a, b) => a - b);
}

export function getCitedSources(
  content: string,
  sources: CitedParent[],
): CitedParent[] {
  return citedDocIndices(content)
    .map((docIndex) => sourceForDocIndex(sources, docIndex))
    .filter((source): source is CitedParent => source !== undefined);
}

export function splitDocCitations(text: string): Array<string | number> {
  const parts: Array<string | number> = [];
  let lastIndex = 0;
  const re = new RegExp(DOC_CITATION_RE.source, "g");
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(Number(match[1]));
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

interface CitationMatch {
  index: number;
  length: number;
  docIndex: number;
}

function findCitationMatches(content: string): CitationMatch[] {
  const matches: CitationMatch[] = [];
  const re = new RegExp(DOC_CITATION_RE.source, "g");
  let match: RegExpExecArray | null;

  while ((match = re.exec(content)) !== null) {
    matches.push({
      index: match.index,
      length: match[0].length,
      docIndex: Number(match[1]),
    });
  }

  return matches;
}

export function dedupeCitationsToLastOccurrence(content: string): string {
  const matches = findCitationMatches(content);
  if (matches.length <= 1) return content;

  const lastIndexByDoc = new Map<number, number>();
  for (const match of matches) {
    lastIndexByDoc.set(match.docIndex, match.index);
  }

  const removable = matches
    .filter((match) => lastIndexByDoc.get(match.docIndex) !== match.index)
    .sort((a, b) => b.index - a.index);

  let result = content;
  for (const match of removable) {
    const before = result.slice(0, match.index);
    const after = result.slice(match.index + match.length);
    result = `${before}${after}`;
  }

  return normalizeCitationPlacement(
    result
      .replace(/[ \t]+(\n|$)/g, "$1")
      .replace(/[ \t]{2,}/g, " ")
      .replace(/ +\./g, ".")
      .replace(/ +,/g, ",")
      .replace(/ +;/g, ";")
      .replace(/ +:/g, ":"),
  );
}

export function normalizeCitationPlacement(content: string): string {
  return content.replace(/\[Doc-(\d+)\]\s*:/g, ": [Doc-$1]");
}

/** Collapse runs like `[Doc-1], [Doc-2]` into a single marker for one chip. */
export function collapseAdjacentCitationRuns(content: string): string {
  return content.replace(
    /(\[Doc-\d+\])(?:[\s,;]+(?:\[Doc-\d+\]))+/g,
    "$1",
  );
}
