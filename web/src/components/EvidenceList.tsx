type Props = { items: string[] };

export function EvidenceList({ items }: Props) {
  if (!items.length) {
    return <div className="muted">no evidence above threshold</div>;
  }
  return (
    <ul className="evidence">
      {items.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ul>
  );
}
