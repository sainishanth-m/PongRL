"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { 
  Gamepad2, 
  BarChart3, 
  Layers, 
  GitCompare, 
  FileText, 
  BookOpen, 
  Menu, 
  X,
  ArrowRight,
  Sparkles
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/gameplay", label: "Play Arena" },
  { href: "/docs", label: "Docs" },
  { href: "/analytics", label: "Analytics" },
  { href: "/comparison", label: "Comparison" },
  { href: "/architecture", label: "Architecture" },
  { href: "/info", label: "FAQ / Viva" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-3 z-50 w-full px-4 sm:px-6">
      <div className="mx-auto max-w-5xl">
        <div className="flex h-14 items-center justify-between rounded-full border border-slate-200/90 bg-white/95 px-5 shadow-[0_4px_24px_rgba(0,0,0,0.06)] backdrop-blur-md transition-all">
          {/* Brand / Logo (matching the [V] icon box + text from the screenshot) */}
          <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-90">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-300 bg-slate-900 text-white font-mono text-xs font-bold shadow-2xs">
              <span className="tracking-tighter">P</span>
            </div>
            <span className="font-bold tracking-tight text-slate-900 text-base">
              PongRL<span className="text-slate-400 font-normal ml-0.5">Pro</span>
            </span>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-1.5 lg:gap-2">
            {NAV_ITEMS.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-1.5 text-xs font-medium transition-all rounded-full ${
                    isActive
                      ? "text-slate-950 font-semibold bg-slate-100"
                      : "text-slate-600 hover:text-slate-950 hover:bg-slate-50"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* Right Action Button (matching the black pill 'Open Workspace' from screenshot) */}
          <div className="flex items-center gap-2">
            <Link
              href="/gameplay"
              className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-slate-950 px-4 py-1.5 text-xs font-semibold text-white transition-all hover:bg-slate-800 shadow-2xs hover:shadow-xs active:scale-[0.98]"
            >
              <span>Play Agent</span>
              <ArrowRight className="h-3.5 w-3.5 text-slate-300" />
            </Link>

            {/* Mobile Hamburger */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 text-slate-700 hover:bg-slate-50"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown */}
        {mobileOpen && (
          <div className="mt-2 md:hidden rounded-2xl border border-slate-200/90 bg-white/98 p-3 shadow-xl backdrop-blur-md animate-in slide-in-from-top-2">
            <div className="flex flex-col gap-1">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className={`flex items-center justify-between rounded-xl px-3.5 py-2.5 text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-slate-100 text-slate-950 font-semibold"
                        : "text-slate-600 hover:bg-slate-50 hover:text-slate-950"
                    }`}
                  >
                    <span>{item.label}</span>
                    {isActive && <span className="h-1.5 w-1.5 rounded-full bg-slate-900" />}
                  </Link>
                );
              })}
              <div className="pt-2 border-t border-slate-100 mt-1">
                <Link
                  href="/gameplay"
                  onClick={() => setMobileOpen(false)}
                  className="flex items-center justify-center gap-2 rounded-xl bg-slate-950 py-2.5 text-sm font-semibold text-white shadow-2xs"
                >
                  <span>Play Live Agent</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
