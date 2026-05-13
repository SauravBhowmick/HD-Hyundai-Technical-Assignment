import { HistoryPoint } from "../types";

type Props = { data: HistoryPoint[] };

type Event = { ts: string; kind: "error" | "maint" | "failure"; label: string };

function collect(data: HistoryPoint[]): Event[] {
  const out: Event[] = [];
  for (const p of data) {
    for (const e of p.errors) out.push({ ts: p.datetime, kind: "error", label: e });
    for (const c of p.maint) out.push({ ts: p.datetime, kind: "maint", label: c });
    for (const f of p.failures) out.push({ ts: p.datetime, kind: "failure", label: f });
  }
  return out;
}

export function EventTimeline({ data }: Props) {
  const events = collect(data);
  if (!events.length) {
    return <div className="muted">no events in this window</div>;
  }
  return (
    <div className="timeline">
      {events.map((e, i) => (
        <div key={i} className="timeline-item">
          <span>{e.ts.replace("T", " ").slice(0, 16)}</span>
          <span className="timeline-kind">{e.kind}</span>
          <span>{e.label}</span>
        </div>
      ))}
    </div>
  );
}
