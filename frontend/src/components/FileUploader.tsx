"use client";

import { useState, useCallback } from "react";
import { Upload, FileImage, Loader2, ShieldCheck, Search } from "lucide-react";
import { motion } from "framer-motion";

interface FileUploaderProps {
    onUpload: (file: File) => void;
    isAnalyzing: boolean;
}

export default function FileUploader({ onUpload, isAnalyzing }: FileUploaderProps) {
    const [isDragging, setIsDragging] = useState(false);

    const handleFile = (file: File | null) => {
        if (file && (file.type === "image/png" || file.type === "image/jpeg" || file.type === "image/jpg")) {
            onUpload(file);
        } else if (file) {
            alert("Please upload a valid image file (PNG, JPG, JPEG)");
        }
    };

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        handleFile(e.dataTransfer.files[0]);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className={`relative group border-[0.5px] rounded-[2rem] p-12 transition-all duration-500 overflow-hidden
        ${isDragging ? "border-[#ff7e5f] bg-[#ff7e5f]/5" : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20"}
        ${isAnalyzing ? "opacity-50 pointer-events-none" : "cursor-pointer"}
        shadow-2xl
      `}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
        >
            {/* Decorative gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-transparent to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

            <input
                id="fileInput"
                type="file"
                className="hidden"
                accept=".png,.jpg,.jpeg"
                onChange={(e) => handleFile(e.target.files?.[0] || null)}
            />

            <div className="relative z-10 flex flex-col items-center gap-6">
                {isAnalyzing ? (
                    <>
                        <div className="relative">
                            <Loader2 className="w-20 h-20 text-[#ff7e5f] animate-spin" />
                            <Search className="absolute inset-0 m-auto w-8 h-8 text-white animate-pulse" />
                        </div>
                        <div className="space-y-2 text-center">
                            <p className="text-2xl font-semibold text-white tracking-tight">Neural Deep Scan in Progress</p>
                            <p className="text-neutral-500 max-w-xs mx-auto">Analyzing voxel gradients and identifying pathological markers...</p>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="relative bg-white/5 p-8 rounded-full border border-white/10 group-hover:scale-105 transition-transform duration-500">
                            <Upload className="w-10 h-10 text-white" />
                            <div className="absolute -top-1 -right-1 flex h-4 w-4">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#ff7e5f] opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-4 w-4 bg-[#ff7e5f]"></span>
                            </div>
                        </div>
                        <div className="space-y-4 text-center">
                            <div>
                                <p className="text-3xl font-bold text-white tracking-tight mb-2">Import Diagnostic Image</p>
                                <p className="text-neutral-400">Drag or click to select chest radiography</p>
                            </div>
                            <div className="flex items-center justify-center gap-4 pt-4">
                                <div className="flex items-center gap-1.5 px-3 py-1 bg-white/5 rounded-full border border-white/5 text-[10px] text-neutral-500 uppercase tracking-widest">
                                    <ShieldCheck className="w-3 h-3" /> HIPAA Compliant
                                </div>
                                <div className="flex items-center gap-1.5 px-3 py-1 bg-white/5 rounded-full border border-white/5 text-[10px] text-neutral-500 uppercase tracking-widest">
                                    Encrypted
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </motion.div>
    );
}
