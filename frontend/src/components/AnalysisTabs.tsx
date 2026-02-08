"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { AnalysisData } from "@/lib/types";
import { Image as ImageIcon, Box, BarChart3, Maximize } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface AnalysisTabsProps {
    data: AnalysisData;
    showHistogram: boolean;
    showStatistics: boolean;
}

export default function AnalysisTabs({ data, showHistogram, showStatistics }: AnalysisTabsProps) {
    const [activeTab, setActiveTab] = useState(0);

    const tabs = [
        { label: "Radiology", icon: <ImageIcon className="w-4 h-4" /> },
        { label: "3D Volumetric", icon: <Box className="w-4 h-4" /> },
        { label: "Metrics", icon: <BarChart3 className="w-4 h-4" /> },
        { label: "Multi-View", icon: <Maximize className="w-4 h-4" /> },
    ];

    return (
        <div className="space-y-8">
            {/* Custom Tab Switcher */}
            <div className="flex gap-2 p-1.5 bg-white/5 border border-white/10 rounded-2xl w-fit mx-auto">
                {tabs.map((tab, idx) => (
                    <button
                        key={idx}
                        onClick={() => setActiveTab(idx)}
                        className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300
              ${activeTab === idx
                                ? "bg-white text-black shadow-lg"
                                : "text-neutral-500 hover:text-white hover:bg-white/5"}
            `}
                    >
                        {tab.icon}
                        {tab.label}
                    </button>
                ))}
            </div>

            <AnimatePresence mode="wait">
                <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 1.02 }}
                    transition={{ duration: 0.3 }}
                    className="bg-white/[0.02] border border-white/10 rounded-[2.5rem] p-10 overflow-hidden"
                >
                    {activeTab === 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                            <div className="space-y-4">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-[#ff7e5f]">Original Scintigraphy</span>
                                <div className="rounded-3xl overflow-hidden border border-white/5 shadow-2xl">
                                    <img src={data.images.original} alt="Scan" className="w-full grayscale hover:grayscale-0 transition-all duration-700" />
                                </div>
                            </div>
                            <div className="space-y-4">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-[#ff7e5f]">Saliency Map</span>
                                <div className="rounded-3xl overflow-hidden border border-white/5 shadow-2xl">
                                    <img src={data.images.overlay} alt="Heatmap" className="w-full" />
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 1 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                            <div className="bg-black/20 p-6 rounded-3xl border border-white/5">
                                <Plot
                                    data={data.visualizations.plot3d_original.data}
                                    layout={{
                                        ...data.visualizations.plot3d_original.layout,
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        font: { color: '#888', family: 'Inter' },
                                        margin: { t: 0, b: 0, l: 0, r: 0 }
                                    }}
                                    useResizeHandler={true}
                                    className="w-full h-[500px]"
                                />
                            </div>
                            <div className="bg-black/20 p-6 rounded-3xl border border-white/5">
                                <Plot
                                    data={data.visualizations.plot3d_overlay.data}
                                    layout={{
                                        ...data.visualizations.plot3d_overlay.layout,
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        font: { color: '#888', family: 'Inter' },
                                        margin: { t: 0, b: 0, l: 0, r: 0 }
                                    }}
                                    useResizeHandler={true}
                                    className="w-full h-[500px]"
                                />
                            </div>
                        </div>
                    )}

                    {activeTab === 2 && (
                        <div className="space-y-12">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                                {showStatistics && [
                                    { label: "Input Statistics", stats: data.statistics.original },
                                    { label: "Heatmap Profile", stats: data.statistics.heatmap }
                                ].map((card, i) => (
                                    <div key={i} className="space-y-4">
                                        <h4 className="text-sm font-bold text-white/40 uppercase tracking-[0.3em]">{card.label}</h4>
                                        <div className="grid grid-cols-2 gap-4">
                                            {Object.entries(card.stats).map(([k, v]) => (
                                                <div key={k} className="p-4 bg-white/5 rounded-2xl border border-white/5">
                                                    <p className="text-[10px] text-neutral-600 uppercase mb-1">{k}</p>
                                                    <p className="text-lg font-bold text-white tracking-tight">{v.toFixed(3)}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                            {showHistogram && (
                                <div className="p-10 border border-dashed border-white/10 rounded-3xl text-center">
                                    <p className="text-neutral-500 text-sm">Interactive histograms are embedded in the 3D analytical engine for realtime inspection.</p>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === 3 && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            {[
                                { l: "Raw Grayscale", src: data.images.original, f: "grayscale" },
                                { l: "Feature Extraction", src: data.images.edges, f: "invert opacity-70" },
                                { l: "Composite Visualization", src: data.images.overlay, f: "" }
                            ].map((item, i) => (
                                <div key={i} className="space-y-4">
                                    <p className="text-[10px] font-bold text-neutral-600 uppercase tracking-widest text-center">{item.l}</p>
                                    <img src={item.src} className={`rounded-2xl border border-white/5 ${item.f}`} alt="view" />
                                </div>
                            ))}
                        </div>
                    )}
                </motion.div>
            </AnimatePresence>
        </div>
    );
}
