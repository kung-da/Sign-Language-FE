import { Menu, X } from "lucide-react";
import { useState } from "react";
import { Button } from "../ui/Button";

const links = ["Home", "Demo", "Pipeline", "Architecture", "Metrics", "Features", "About"];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-background/80 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <a href="#home" className="flex items-center gap-3 font-bold text-text">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-cyan text-slate-950">V</span>
          <span>VSL AI Translator</span>
        </a>
        <div className="hidden items-center gap-6 lg:flex">
          {links.map((link) => (
            <a key={link} href={`#${link.toLowerCase()}`} className="text-sm text-muted hover:text-text">
              {link}
            </a>
          ))}
          <Button onClick={() => document.querySelector("#demo")?.scrollIntoView()}>
            Try Demo
          </Button>
        </div>
        <button
          className="rounded-lg p-2 text-text hover:bg-white/10 lg:hidden"
          onClick={() => setOpen((value) => !value)}
          aria-label="Toggle navigation"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </nav>
      {open && (
        <div className="border-t border-white/10 px-4 py-4 lg:hidden">
          <div className="mx-auto grid max-w-7xl gap-3">
            {links.map((link) => (
              <a
                key={link}
                href={`#${link.toLowerCase()}`}
                className="rounded-lg px-3 py-2 text-sm text-muted hover:bg-white/10 hover:text-text"
                onClick={() => setOpen(false)}
              >
                {link}
              </a>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
