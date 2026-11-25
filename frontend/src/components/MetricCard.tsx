import React from "react";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  subtitle?: string;
  icon?: LucideIcon;
}

export default function MetricCard({
  label,
  value,
  change,
  changeType = "positive",
  subtitle,
  icon: Icon
}: MetricCardProps) {
  const badgeClasses = {
    positive: "bg-emerald-50 text-emerald-700 border-emerald-200/60",
    negative: "bg-rose-50 text-rose-700 border-rose-200/60",
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
  }[changeType];

  return (
    <div className="card-minimal p-5 flex flex-col justify-between transition-all">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          {label}
        </span>
        {Icon && (
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-50 border border-slate-100 text-slate-500">
            <Icon className="h-4 w-4" />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-baseline gap-2.5">
        <span className="text-3xl font-bold tracking-tight text-slate-900">
          {value}
        </span>
        {change && (
          <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold ${badgeClasses}`}>
            {change}
          </span>
        )}
      </div>

      {subtitle && (
        <p className="mt-2 text-xs text-slate-500 font-medium leading-relaxed">
          {subtitle}
        </p>
      )}
    </div>
  );
}
