"use client";

import React, { useState } from "react";
import { 
  Info, 
  BookOpen, 
  CheckCircle, 
  HelpCircle, 
  ChevronDown, 
  ChevronUp, 
  FileText, 
  Award,
  ExternalLink,
  GraduationCap
} from "lucide-react";
import { PROJECT_INFO_DATA } from "@/data/projectInfoData";

export default function ProjectInfoPage() {
  const [openVivaIndex, setOpenVivaIndex] = useState<number | null>(0);

  const toggleViva = (index: number) => {
    setOpenVivaIndex(openVivaIndex === index ? null : index);
  };

  return (
    <div className="space-y-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-xs font-semibold mb-3">
          <BookOpen className="w-3.5 h-3.5 text-blue-600" />
          <span>Academic Thesis & Capstone Defense</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Project Information & Defense Dossier
        </h1>
        <p className="text-slate-500 mt-2 text-base max-w-3xl">
          Comprehensive academic documentation detailing the formal problem formulation,
          research methodology, IEEE references, and viva voce defense questions for PongRL Pro.
        </p>
      </div>

      {/* Problem Statement Card */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <div className="flex items-center gap-2 text-blue-600 font-semibold text-xs uppercase tracking-wider mb-2">
          <FileText className="w-4 h-4" />
          <span>Formal Problem Statement</span>
        </div>
        <h2 className="text-xl font-bold text-slate-900 mb-3">
          Vision-Based Policy Acquisition in Sparse Reward Domains
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          {PROJECT_INFO_DATA.problemStatement}
        </p>
      </div>

      {/* Objectives */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Award className="w-5 h-5 text-blue-600" />
          <span>Key Project Objectives</span>
        </h2>
        <div className="space-y-3">
          {PROJECT_INFO_DATA.objectives.map((obj, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100">
              <CheckCircle className="w-4 h-4 text-emerald-600 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-slate-700 leading-relaxed font-medium">{obj}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Methodology Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {PROJECT_INFO_DATA.methodology.map((meth, idx) => (
          <div key={idx} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
            <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">Method 0{idx + 1}</span>
            <h3 className="text-base font-bold text-slate-900 mt-1 mb-2">{meth.title}</h3>
            <p className="text-xs text-slate-600 leading-relaxed">{meth.desc}</p>
          </div>
        ))}
      </div>

      {/* Viva Voce Defense Q&A Accordion */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <div className="flex items-center gap-2 text-purple-600 font-semibold text-xs uppercase tracking-wider mb-2">
          <GraduationCap className="w-4 h-4" />
          <span>Capstone Examination Preparation</span>
        </div>
        <h2 className="text-lg font-bold text-slate-900 mb-2">
          Viva Voce Defense Questions & Answers
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Rigorous technical explanations for common queries raised by university project evaluators:
        </p>

        <div className="space-y-3">
          {PROJECT_INFO_DATA.vivaQuestions.map((item, idx) => {
            const isOpen = openVivaIndex === idx;
            return (
              <div
                key={idx}
                className="border border-slate-200 rounded-xl overflow-hidden transition-all bg-white"
              >
                <button
                  onClick={() => toggleViva(idx)}
                  className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-50 transition-colors"
                >
                  <span className="text-sm font-semibold text-slate-900 pr-4">
                    Q{idx + 1}: {item.q}
                  </span>
                  {isOpen ? (
                    <ChevronUp className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  )}
                </button>
                {isOpen && (
                  <div className="p-4 pt-0 text-xs text-slate-600 leading-relaxed border-t border-slate-100 bg-slate-50/50">
                    <p className="pt-3 font-normal text-slate-700">{item.a}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* IEEE References */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <h2 className="text-base font-bold text-slate-900 mb-4 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-slate-600" />
          <span>Academic References & Citations</span>
        </h2>
        <div className="divide-y divide-slate-100">
          {PROJECT_INFO_DATA.references.map((ref) => (
            <div key={ref.id} className="py-2.5 flex items-start gap-3 text-xs text-slate-600">
              <span className="font-mono text-slate-400 font-semibold flex-shrink-0">[{ref.id}]</span>
              <p className="leading-relaxed">{ref.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
