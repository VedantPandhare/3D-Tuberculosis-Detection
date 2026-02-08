export interface AnalysisData {
    results: {
        prediction: string;
        confidence: number;
        risk_level: string;
    };
    images: {
        original: string;
        overlay: string;
        edges: string;
    };
    visualizations: {
        plot3d_original: any;
        plot3d_overlay: any;
    };
    statistics: {
        original: {
            Mean: number;
            "Std Dev": number;
            Min: number;
            Max: number;
            Median: number;
        };
        heatmap: {
            Mean: number;
            "Std Dev": number;
            Min: number;
            Max: number;
            Median: number;
        };
    };
    metadata: {
        filename: string;
        timestamp: string;
    };
}
