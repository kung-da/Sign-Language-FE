import { X } from "lucide-react";
import { useState } from "react";
import { vectorDbService } from "../../services/vectorDbService";
import { Button } from "../ui/Button";

interface AddNewSignModalProps {
  open: boolean;
  onClose: () => void;
}

export function AddNewSignModal({ open, onClose }: AddNewSignModalProps) {
  const [label, setLabel] = useState("");
  const [description, setDescription] = useState("");
  const [region, setRegion] = useState("");
  const [saved, setSaved] = useState(false);

  if (!open) return null;

  const save = async () => {
    await vectorDbService.saveNewSign({
      label,
      description,
      region,
      confidence: 0,
    });
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      setLabel("");
      setDescription("");
      setRegion("");
      onClose();
    }, 900);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/75 px-4 backdrop-blur-sm">
      <div className="glass-card w-full max-w-lg rounded-lg p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-text">Add New Sign</h3>
            <p className="mt-1 text-sm text-muted">Mock save only. This prepares the payload for a future vector DB.</p>
          </div>
          <button className="rounded-lg p-2 text-muted hover:bg-white/10 hover:text-text" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>
        <div className="mt-5 space-y-4">
          <label className="block">
            <span className="text-sm font-medium text-text">Vietnamese text / gloss</span>
            <input
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              className="mt-2 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-text outline-none focus:border-cyan"
              placeholder="Example: Xin chao"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-text">Description</span>
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              className="mt-2 min-h-24 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-text outline-none focus:border-cyan"
              placeholder="Describe hand shape, movement, and context."
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-text">Region / dialect</span>
            <input
              value={region}
              onChange={(event) => setRegion(event.target.value)}
              className="mt-2 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-text outline-none focus:border-cyan"
              placeholder="Optional"
            />
          </label>
        </div>
        {saved && <p className="mt-4 text-sm text-success">Saved to mock vector payload.</p>}
        <div className="mt-6 flex justify-end gap-3">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!label.trim()}>
            Save
          </Button>
        </div>
      </div>
    </div>
  );
}
