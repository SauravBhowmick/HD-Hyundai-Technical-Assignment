type Props = {
  value: number;
  threshold?: number;
};

export function RiskGauge({ value, threshold }: Props) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div className="gauge-wrap">
      <div className="gauge-bar" title={threshold ? `threshold ${threshold.toFixed(2)}` : undefined}>
        <div className="gauge-fill" style={{ width: `${pct * 100}%` }} />
      </div>
      <div className="gauge-number">{(pct * 100).toFixed(1)}%</div>
    </div>
  );
}
