"use client";

import React, { useState } from "react";
import { 
  Sliders, 
  Search, 
  Copy, 
  Check, 
  Download, 
  FileCode, 
  Settings2,
  RefreshCw,
  Sparkles
} from "lucide-react";
import { HYPERPARAMETERS_DATA } from "@/data/hyperparamsData";

export default function HyperparametersPage() {
  const [selectedGroup, setSelectedGroup] = useState<string>("All");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);

  // Filter groups
  const groups = ["All", ...HYPERPARAMETERS_DATA.map((g) => g.name)];

  const filteredGroups = HYPERPARAMETERS_DATA.filter((group) => {
    if (selectedGroup !== "All" && group.name !== selectedGroup) return false;
    return true;
  }).map((group) => {
    if (!searchQuery.trim()) return group;
    const q = searchQuery.toLowerCase();
    const filteredParams = group.params.filter(
      (p) =>
        p.key.toLowerCase().includes(q) ||
        p.label.toLowerCase().includes(q) ||
        p.desc.toLowerCase().includes(q)
    );
    return { ...group, params: filteredParams };
  }).filter((group) => group.params.length > 0);

  // Generate YAML representation
  const generateYaml = () => {
    let yaml = `# PongRL Pro Hyperparameter Specifications\n# Auto-generated from project runtime configurations\n\n`;
    HYPERPARAMETERS_DATA.forEach((group) => {
      const slug = group.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
      yaml += `${slug}:\n`;
      group.params.forEach((p) => {
        yaml += `  ${p.key}: ${typeof p.value === "string" ? `"${p.value}"` : p.value}  # ${p.desc}\n`;
      });
      yaml += `\n`;
    });
    return yaml;
  };

  const handleCopyYaml = () => {
    navigator.clipboard.writeText(generateYaml());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadYaml = () => {
    const blob = new Blob([generateYaml()], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pongrl_hyperparameters.yaml";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-10 max-w-6xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-xs font-semibold mb-3">
          <Sliders className="w-3.5 h-3.5 text-blue-600" />
          <span>Empirical Parameterization</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Hyperparameter Configuration
        </h1>
        <p className="text-slate-500 mt-2 text-base max-w-3xl">
          Complete parameter inventory extracted from the actual YAML configuration files across
          environment preprocessors, REINFORCE policy gradients, DQN value approximations, and PPO rollouts.
        </p>

        {/* Search & Actions Bar */}
        <div className="flex flex-col sm:flex-row gap-4 mt-6 items-stretch sm:items-center justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search parameters by key, label, or rationale..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 transition-all bg-white"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleCopyYaml}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-colors shadow-2xs"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-500" />}
              <span>{copied ? "Copied" : "Copy YAML"}</span>
            </button>
            <button
              onClick={handleDownloadYaml}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-2xs"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download .yaml</span>
            </button>
          </div>
        </div>

        {/* Group Selector Pills */}
        <div className="flex flex-wrap gap-2 mt-4">
          {groups.map((group) => {
            const isSelected = selectedGroup === group;
            return (
              <button
                key={group}
                onClick={() => setSelectedGroup(group)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
                  isSelected
                    ? "bg-slate-900 border-slate-900 text-white shadow-2xs"
                    : "bg-white border-slate-200 text-slate-600 hover:border-slate-300 hover:text-slate-900"
                }`}
              >
                {group}
              </button>
            );
          })}
        </div>
      </div>

      {/* Parameter Tables */}
      <div className="space-y-8">
        {filteredGroups.length === 0 ? (
          <div className="text-center py-12 bg-white border border-slate-200 rounded-2xl">
            <p className="text-sm text-slate-500">No parameters found matching &quot;{searchQuery}&quot;.</p>
          </div>
        ) : (
          filteredGroups.map((group, gIdx) => (
            <div key={gIdx} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="mb-4">
                <h2 className="text-base font-semibold text-slate-900">{group.name}</h2>
                <p className="text-xs text-slate-500 mt-0.5">{group.description}</p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase">
                      <th className="py-2.5 px-4">Parameter Key</th>
                      <th className="py-2.5 px-4">Display Label</th>
                      <th className="py-2.5 px-4">Value</th>
                      <th className="py-2.5 px-4">Type</th>
                      <th className="py-2.5 px-4">Description & Empirical Rationale</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {group.params.map((param, pIdx) => (
                      <tr key={pIdx} className="hover:bg-slate-50/70 transition-colors">
                        <td className="py-3 px-4 font-mono text-xs font-semibold text-slate-900">
                          {param.key}
                        </td>
                        <td className="py-3 px-4 text-xs font-medium text-slate-800">
                          {param.label}
                        </td>
                        <td className="py-3 px-4 font-mono text-xs text-blue-600 font-bold">
                          {String(param.value)}
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                            {param.type}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-xs text-slate-600 max-w-md leading-relaxed">
                          {param.desc}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Embedded YAML Preview */}
      <div className="bg-slate-900 text-slate-200 rounded-2xl p-6 shadow-xs font-mono text-xs space-y-3">
        <div className="flex items-center justify-between text-slate-400 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <FileCode className="w-4 h-4 text-blue-400" />
            <span className="font-bold text-slate-300">config/default.yaml</span>
          </div>
          <span>Synchronized with PyTorch Trainer</span>
        </div>
        <pre className="text-slate-300 overflow-x-auto max-h-72 overflow-y-auto leading-relaxed">
          {generateYaml()}
        </pre>
      </div>
    </div>
  );
}
