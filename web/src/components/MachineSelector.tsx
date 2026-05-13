import { MachineInfo } from "../types";

type Props = {
  machines: MachineInfo[];
  value: number | null;
  onChange: (mid: number) => void;
};

export function MachineSelector({ machines, value, onChange }: Props) {
  return (
    <div className="field">
      <label htmlFor="machine">Machine</label>
      <select
        id="machine"
        value={value ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        <option value="" disabled>
          Select a machine
        </option>
        {machines.map((m) => (
          <option key={m.machineID} value={m.machineID}>
            #{m.machineID} - {m.model} - age {m.age_years}y
          </option>
        ))}
      </select>
    </div>
  );
}
