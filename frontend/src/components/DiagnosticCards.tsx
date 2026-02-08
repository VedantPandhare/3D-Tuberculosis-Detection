"use client";

import { Activity, Percent, ShieldCheck, AlertCircle, Info } from "lucide-react";
import { AnalysisData } from "@/lib/types";
import { motion } from "framer-motion";

interface DiagnosticCardsProps {
    results: AnalysisData['results'];
}

export default function DiagnosticCards({ results }: DiagnosticCardsProps) {
    const { prediction, confidence, risk_level } = results;
    const isTB = prediction === "TB Detected";

    const cards = [
        {
            title: "Classification",
            value: prediction,
            icon: <Activity className="w-5 h-5" />,
            highlight: isTB ? "text-[#ff7e5f]" : "text-emerald-400",
            bg: "bg-white/[0.03]",
        },
        {
            title: "Confidence",
            value: `${confidence}%`,
            icon: <Percent className="w-5 h-5" />,
            highlight: "text-white",
            bg: "bg-white/[0.03]",
        },
        {
            title: "Risk Profile",
            value: risk_level,
            icon: <ShieldCheck className="w-5 h-5" />,
            highlight: risk_level === "HIGH" ? "text-red-500" : risk_level === "MODERATE" ? "text-yellow-500" : "text-emerald-500",
            bg: "bg-white/[0.03]",
        }
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {cards.map((card, idx) => (
                <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className={`relative p-8 rounded-[2rem] border border-white/10 ${card.bg} overflow-hidden group`}
                >
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        {card.icon}
                    </div>
                    <div className="flex flex-col gap-4 relative z-10">
                        <span className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">
                            {card.title}
                        </span>
                        <p className={`text-4xl font-extrabold tracking-tighter ${card.highlight}`}>
                            {card.value}
                        </p>
                    </div>

                    <div className={`absolute bottom-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-white/10 to-transparent scale-x-0 group-hover:scale-x-100 transition-transform duration-700`} />
                </motion.div>
            ))}
        </div>
    );
}
