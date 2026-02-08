"use client";
import React, { useEffect, useRef, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import { cn } from "@/lib/utils";

export const BackgroundRippleEffect = ({
    className,
    containerClassName,
}: {
    className?: string;
    containerClassName?: string;
}) => {
    const [rows, setRows] = useState(0);
    const [cols, setCols] = useState(0);

    useEffect(() => {
        const updateGrid = () => {
            setRows(Math.ceil(window.innerHeight / 50));
            setCols(Math.ceil(window.innerWidth / 50));
        };
        updateGrid();
        window.addEventListener("resize", updateGrid);
        return () => window.removeEventListener("resize", updateGrid);
    }, []);

    return (
        <div
            className={cn(
                "absolute inset-0 z-0 flex h-full w-full flex-wrap justify-center overflow-hidden bg-[#0a0a0a]",
                containerClassName
            )}
        >
            <div
                className="grid h-full w-full"
                style={{
                    gridTemplateColumns: `repeat(${cols}, 1fr)`,
                    gridTemplateRows: `repeat(${rows}, 1fr)`,
                }}
            >
                {Array.from({ length: rows * cols }).map((_, i) => (
                    <RippleBox key={i} className={className} />
                ))}
            </div>
        </div>
    );
};

const RippleBox = ({ className }: { className?: string }) => {
    const controls = useAnimation();

    const handleInteraction = async () => {
        await controls.start({
            backgroundColor: "rgba(255, 126, 95, 0.4)",
            transition: { duration: 0.1 },
        });
        await controls.start({
            backgroundColor: "rgba(255, 255, 255, 0)",
            transition: { duration: 0.8 },
        });
    };

    return (
        <motion.div
            animate={controls}
            onMouseEnter={handleInteraction}
            onClick={handleInteraction}
            className={cn(
                "h-full w-full border-[0.5px] border-white/5 transition-colors duration-500",
                className
            )}
        />
    );
};
