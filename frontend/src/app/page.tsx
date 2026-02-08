"use client";

import { useState } from "react";
import { BackgroundRippleEffect } from "@/components/ui/background-ripple-effect";
import SettingsPanel from "../components/SettingsPanel";
import FileUploader from "@/components/FileUploader";
import DiagnosticCards from "@/components/DiagnosticCards";
import AnalysisTabs from "@/components/AnalysisTabs";
import ClinicalNotes from "@/components/ClinicalNotes";
import { AnalysisData } from "@/lib/types";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, Settings as SettingsIcon, Activity } from "lucide-react";

export default function Dashboard() {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLanding, setIsLanding] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Settings state
  const [heatmapIntensity, setHeatmapIntensity] = useState(0.4);
  const [showHistogram, setShowHistogram] = useState(true);
  const [showStatistics, setShowStatistics] = useState(true);

  const handleAnalysis = async (file: File) => {
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Analysis failed");

      const data = await response.json();
      setAnalysisData(data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to analyze image. Please verify backend.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-[#0a0a0a] text-[#ededed] overflow-x-hidden">
      <BackgroundRippleEffect />

      {/* Slide-in settings panel */}
      <AnimatePresence>
        {isSidebarOpen && (
          <SettingsPanel
            onClose={() => setIsSidebarOpen(false)}
            heatmapIntensity={heatmapIntensity}
            setHeatmapIntensity={setHeatmapIntensity}
            showHistogram={showHistogram}
            setShowHistogram={setShowHistogram}
            showStatistics={showStatistics}
            setShowStatistics={setShowStatistics}
          />
        )}
      </AnimatePresence>

      <div className="relative z-10 font-sans">
        <AnimatePresence mode="wait">
          {isLanding ? (
            /* GET STARTED PAGE */
            <motion.section
              key="landing"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center justify-center min-h-screen p-6 text-center"
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="mb-8 p-4 bg-white/5 rounded-full border border-white/10 backdrop-blur-md"
              >
                <Activity className="w-16 h-16 text-[#ff7e5f]" />
              </motion.div>

              <h1 className="max-w-4xl text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-white via-white to-gray-500 bg-clip-text text-transparent">
                Precision Tuberculosis <br /> Detection via AI
              </h1>

              <p className="max-w-2xl text-lg md:text-xl text-neutral-400 mb-12 leading-relaxed">
                Experience world-class diagnostic intelligence. Our advanced deep learning model
                provides instantaneous analysis with clinical-grade accuracy and state-of-the-art
                explainable AI visualizations.
              </p>

              <button
                onClick={() => setIsLanding(false)}
                className="group relative flex items-center gap-2 px-8 py-4 bg-white text-black font-bold rounded-full overflow-hidden transition-all hover:pr-10 active:scale-95 shadow-[0_0_40px_rgba(255,255,255,0.2)]"
              >
                Get Started
                <ChevronRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
              </button>
            </motion.section>
          ) : (
            /* MAIN UPLOAD & ANALYSIS PAGE */
            <motion.section
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-7xl mx-auto px-6 py-12"
            >
              {/* Header with settings trigger */}
              <div className="flex justify-between items-center mb-16">
                <div>
                  <h2 className="text-3xl font-bold text-white">Analysis Hub</h2>
                  <p className="text-neutral-500">Upload medical imagery for evaluation</p>
                </div>
                <button
                  onClick={() => setIsSidebarOpen(true)}
                  className="p-3 bg-white/5 rounded-2xl border border-white/10 hover:bg-white/10 transition-colors"
                >
                  <SettingsIcon className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 gap-12">
                <div className="space-y-4">
                  <div className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold mb-2">Instructions</h3>
                    <ul className="text-neutral-400 space-y-2 list-disc list-inside text-sm">
                      <li>Ensure X-ray is clear and properly oriented.</li>
                      <li>Supported formats: DICOM, PNG, JPG (Standard resolution).</li>
                      <li>Wait for the neural network to process voxel data.</li>
                    </ul>
                  </div>

                  <FileUploader
                    onUpload={handleAnalysis}
                    isAnalyzing={isAnalyzing}
                  />
                </div>

                {analysisData && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-12 pb-20"
                  >
                    <DiagnosticCards results={analysisData.results} />
                    <AnalysisTabs
                      data={analysisData}
                      showHistogram={showHistogram}
                      showStatistics={showStatistics}
                    />
                    <ClinicalNotes data={analysisData} />
                  </motion.div>
                )}
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
