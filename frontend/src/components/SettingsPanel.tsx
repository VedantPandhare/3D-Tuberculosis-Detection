"use client";

import { motion } from "framer-motion";
import { X, SlidersHorizontal, Info, BarChart3 } from "lucide-react";

interface SettingsPanelProps {
    onClose: () => void;
    heatmapIntensity: number;
    setHeatmapIntensity: (val: number) => void;
    showHistogram: boolean;
    setShowHistogram: (val: boolean) => void;
    showStatistics: boolean;
    setShowStatistics: (val: boolean) => void;
}

export default function SettingsPanel({
    onClose,
    heatmapIntensity,
    setHeatmapIntensity,
    showHistogram,
    setShowHistogram,
    showStatistics,
    setShowStatistics,
}: SettingsPanelProps) {
    return (
        <>
            {/* Backdrop */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={onClose}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            />

            {/* Panel */}
            <motion.aside
                initial={{ x: "100%" }}
                animate={{ x: 0 }}
                exit={{ x: "100%" }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="fixed top-0 right-0 h-full w-full max-w-sm bg-[#111] border-l border-white/10 p-8 z-[60] shadow-2xl overflow-y-auto"
            >
                <div className="flex justify-between items-center mb-10">
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                        <SlidersHorizontal className="w-6 h-6 text-[#ff7e5f]" />
                        Settings
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-white/5 rounded-full transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="space-y-10">
                    <section className="space-y-6">
                        <h3 className="text-sm font-semibold uppercase tracking-widest text-neutral-500">Visualization</h3>

                        <div className="space-y-4">
                            <div className="flex justify-between text-sm">
                                <label className="text-neutral-400">Heatmap Intensity</label>
                                <span className="text-[#ff7e5f] font-mono">{heatmapIntensity}</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={heatmapIntensity}
                                onChange={(e) => setHeatmapIntensity(parseFloat(e.target.value))}
                                className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-[#ff7e5f]"
                            />
                        </div>

                        <div className="flex flex-col gap-4">
                            <label className="flex items-center gap-3 cursor-pointer group">
                                <div className={`w-5 h-5 rounded border transition-all flex items-center justify-center ${showHistogram ? 'bg-[#ff7e5f] border-[#ff7e5f]' : 'border-neutral-700'}`}>
                                    {showHistogram && <div className="w-2 h-2 bg-white rounded-full" />}
                                </div>
                                <input
                                    type="checkbox"
                                    className="hidden"
                                    checked={showHistogram}
                                    onChange={(e) => setShowHistogram(e.target.checked)}
                                />
                                <span className="text-neutral-300 group-hover:text-white transition-colors">Show Intensity Histogram</span>
                            </label>

                            <label className="flex items-center gap-3 cursor-pointer group">
                                <div className={`w-5 h-5 rounded border transition-all flex items-center justify-center ${showStatistics ? 'bg-[#ff7e5f] border-[#ff7e5f]' : 'border-neutral-700'}`}>
                                    {showStatistics && <div className="w-2 h-2 bg-white rounded-full" />}
                                </div>
                                <input
                                    type="checkbox"
                                    className="hidden"
                                    checked={showStatistics}
                                    onChange={(e) => setShowStatistics(e.target.checked)}
                                />
                                <span className="text-neutral-300 group-hover:text-white transition-colors">Show Statistics</span>
                            </label>
                        </div>
                    </section>

                    <section className="space-y-4">
                        <h3 className="text-sm font-semibold uppercase tracking-widest text-neutral-500">System Info</h3>
                        <div className="p-4 bg-white/5 rounded-2xl border border-white/5 space-y-3 text-sm text-neutral-400">
                            <div className="flex items-start gap-3">
                                <Info className="w-4 h-4 mt-0.5 text-[#ff7e5f]" />
                                <p>Model: VersaNet-TB-v4 (Proprietary CNN Architecture)</p>
                            </div>
                            <div className="flex items-start gap-3">
                                <BarChart3 className="w-4 h-4 mt-0.5 text-[#ff7e5f]" />
                                <p>Confidence Floor: 0.82 (Validated Dataset)</p>
                            </div>
                        </div>
                    </section>
                </div>

                <div className="mt-20 pt-8 border-t border-white/5 text-center">
                    <p className="text-xs text-neutral-600 font-mono">BUILD 2026.02.08.104</p>
                </div>
            </motion.aside>
        </>
    );
}
