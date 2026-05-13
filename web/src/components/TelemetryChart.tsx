import { HistoryPoint } from "../types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Props = { data: HistoryPoint[] };

const SERIES: { key: keyof HistoryPoint; color: string; label: string }[] = [
  { key: "volt", color: "#6aa9ff", label: "volt" },
  { key: "rotate", color: "#2ecc71", label: "rotate" },
  { key: "pressure", color: "#f1c40f", label: "pressure" },
  { key: "vibration", color: "#e74c3c", label: "vibration" },
];

export function TelemetryChart({ data }: Props) {
  if (!data.length) {
    return <div className="muted">no telemetry in this window</div>;
  }
  const chartData = data
    .filter((p) => p.volt != null || p.rotate != null)
    .map((p) => ({
      t: p.datetime.slice(5, 16),
      volt: p.volt,
      rotate: p.rotate,
      pressure: p.pressure,
      vibration: p.vibration,
    }));
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#243056" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#8a93b3" }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10, fill: "#8a93b3" }} />
          <Tooltip
            contentStyle={{
              background: "#131a30",
              border: "1px solid #243056",
              borderRadius: 8,
              color: "#e8ecf7",
            }}
          />
          {SERIES.map((s) => (
            <Line
              key={s.key as string}
              type="monotone"
              dataKey={s.key as string}
              stroke={s.color}
              strokeWidth={1.5}
              dot={false}
              name={s.label}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
