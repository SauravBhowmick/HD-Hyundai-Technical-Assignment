type Props = {
  value: string;
  min: string;
  max: string;
  onChange: (ts: string) => void;
};

function trimSeconds(s: string): string {
  return s.length >= 16 ? s.slice(0, 16) : s;
}

export function TimestampPicker({ value, min, max, onChange }: Props) {
  return (
    <div className="field">
      <label htmlFor="ts">Timestamp</label>
      <input
        id="ts"
        type="datetime-local"
        min={trimSeconds(min)}
        max={trimSeconds(max)}
        value={trimSeconds(value)}
        onChange={(e) => onChange(`${e.target.value}:00`)}
      />
      <div className="muted" style={{ fontSize: 11 }}>
        range: {trimSeconds(min)} - {trimSeconds(max)}
      </div>
    </div>
  );
}
