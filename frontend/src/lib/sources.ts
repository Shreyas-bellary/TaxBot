import { metaNumber, metaString, type CitedParent } from "./types";

export function sourceTitle(source: CitedParent): string {
  return (
    metaString(source.metadata, "doc_title") ??
    metaString(source.metadata, "doc_number") ??
    "IRS document"
  );
}

export function sourceBadges(source: CitedParent): string[] {
  const badges: string[] = [];
  const docNumber = metaString(source.metadata, "doc_number");
  const taxYear = metaNumber(source.metadata, "tax_year");
  const nodeKind = metaString(source.metadata, "node_kind");
  if (docNumber) badges.push(docNumber);
  if (taxYear !== null) badges.push(`Tax year ${taxYear}`);
  if (nodeKind === "table") badges.push("Table");
  return badges;
}
