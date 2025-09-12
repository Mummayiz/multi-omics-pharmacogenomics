/**
 * Multi-Omics Visualization Module
 * Handles chart creation and data visualization
 */

class VisualizationManager {
    constructor() {
        this.charts = {};
    }

    /**
     * Create SHAP values plot
     */
    createShapPlot(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Sample SHAP visualization
        const plotHtml = `
            <div class="shap-plot">
                <h4>Feature Importance (SHAP Values)</h4>
                <div class="feature-bars">
                    ${this.generateFeatureBars(data)}
                </div>
                <p class="plot-description">
                    Features with positive SHAP values increase the prediction, 
                    while negative values decrease it.
                </p>
            </div>
        `;

        container.innerHTML = plotHtml;
    }

    /**
     * Generate feature importance bars
     */
    generateFeatureBars(data) {
        const features = [
            { name: 'EGFR mutation', value: 0.25, type: 'genomic' },
            { name: 'Gene expression signature', value: 0.18, type: 'transcriptomic' },
            { name: 'Protein abundance', value: 0.15, type: 'proteomic' },
            { name: 'KRAS wild-type', value: 0.12, type: 'genomic' },
            { name: 'Pathway activity', value: 0.08, type: 'transcriptomic' }
        ];

        return features.map(feature => {
            const percentage = Math.abs(feature.value) * 100;
            const color = this.getFeatureColor(feature.type);
            const direction = feature.value > 0 ? 'positive' : 'negative';

            return `
                <div class="feature-bar ${direction}">
                    <span class="feature-name">${feature.name}</span>
                    <div class="bar-container">
                        <div class="bar" style="width: ${percentage}%; background-color: ${color};">
                            <span class="bar-value">${feature.value.toFixed(3)}</span>
                        </div>
                    </div>
                    <span class="feature-type">${feature.type}</span>
                </div>
            `;
        }).join('');
    }

    /**
     * Create attention heatmap
     */
    createAttentionHeatmap(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const heatmapHtml = `
            <div class="attention-heatmap">
                <h4>Cross-Omics Attention Weights</h4>
                <div class="heatmap-grid">
                    ${this.generateHeatmapGrid()}
                </div>
                <div class="heatmap-legend">
                    <span>Low</span>
                    <div class="legend-gradient"></div>
                    <span>High</span>
                </div>
                <p class="plot-description">
                    Attention weights show which omics features the model focuses on 
                    when making predictions.
                </p>
            </div>
        `;

        container.innerHTML = heatmapHtml;
    }

    /**
     * Generate heatmap grid
     */
    generateHeatmapGrid() {
        const omicsTypes = ['Genomics', 'Transcriptomics', 'Proteomics'];
        const features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'];
        
        let gridHtml = '<div class="heatmap-header"><div></div>';
        
        // Header row
        omicsTypes.forEach(omics => {
            gridHtml += `<div class="header-cell">${omics}</div>`;
        });
        gridHtml += '</div>';

        // Data rows
        features.forEach(feature => {
            gridHtml += `<div class="heatmap-row"><div class="row-header">${feature}</div>`;
            
            omicsTypes.forEach(() => {
                const intensity = Math.random();
                const color = this.getHeatmapColor(intensity);
                gridHtml += `<div class="heatmap-cell" style="background-color: ${color}" title="Attention: ${intensity.toFixed(3)}"></div>`;
            });
            
            gridHtml += '</div>';
        });

        return gridHtml;
    }

    /**
     * Create feature importance chart
     */
    createFeatureImportanceChart(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Use Chart.js if available, otherwise create simple HTML chart
        if (typeof Chart !== 'undefined') {
            this.createChartJsFeatureImportance(container, data);
        } else {
            this.createHtmlFeatureImportance(container, data);
        }
    }

    /**
     * Create Chart.js feature importance chart
     */
    createChartJsFeatureImportance(container, data) {
        const canvas = document.createElement('canvas');
        container.innerHTML = '';
        container.appendChild(canvas);

        const features = [
            { name: 'EGFR mutation', importance: 0.25, category: 'Genomics' },
            { name: 'Gene signature', importance: 0.18, category: 'Transcriptomics' },
            { name: 'Protein level', importance: 0.15, category: 'Proteomics' },
            { name: 'SNP rs123456', importance: 0.12, category: 'Genomics' },
            { name: 'Pathway score', importance: 0.08, category: 'Transcriptomics' }
        ];

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: features.map(f => f.name),
                datasets: [{
                    label: 'Feature Importance',
                    data: features.map(f => f.importance),
                    backgroundColor: features.map(f => this.getFeatureColor(f.category.toLowerCase())),
                    borderColor: features.map(f => this.getFeatureColor(f.category.toLowerCase())),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance Analysis'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    }
                }
            }
        });
    }

    /**
     * Create HTML-based feature importance chart
     */
    createHtmlFeatureImportance(container, data) {
        const features = [
            { name: 'EGFR mutation', importance: 0.25, category: 'Genomics' },
            { name: 'Gene signature', importance: 0.18, category: 'Transcriptomics' },
            { name: 'Protein level', importance: 0.15, category: 'Proteomics' },
            { name: 'SNP rs123456', importance: 0.12, category: 'Genomics' },
            { name: 'Pathway score', importance: 0.08, category: 'Transcriptomics' }
        ];

        const chartHtml = `
            <div class="feature-importance-chart">
                <h4>Feature Importance Analysis</h4>
                <div class="chart-bars">
                    ${features.map(feature => {
                        const percentage = feature.importance * 100;
                        const color = this.getFeatureColor(feature.category.toLowerCase());
                        
                        return `
                            <div class="chart-bar">
                                <div class="bar-info">
                                    <span class="bar-label">${feature.name}</span>
                                    <span class="bar-category">${feature.category}</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width: ${percentage}%; background-color: ${color};">
                                        <span class="bar-value">${feature.importance.toFixed(3)}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;

        container.innerHTML = chartHtml;
    }

    /**
     * Get color for feature type
     */
    getFeatureColor(type) {
        const colors = {
            'genomic': '#3b82f6',      // Blue
            'genomics': '#3b82f6',     // Blue
            'transcriptomic': '#10b981', // Green
            'transcriptomics': '#10b981', // Green
            'proteomic': '#f59e0b',    // Orange
            'proteomics': '#f59e0b',   // Orange
            'fusion': '#8b5cf6'        // Purple
        };
        return colors[type.toLowerCase()] || '#6b7280';
    }

    /**
     * Get heatmap color based on intensity
     */
    getHeatmapColor(intensity) {
        // Create color gradient from light blue to dark red
        const r = Math.floor(255 * intensity);
        const g = Math.floor(255 * (1 - intensity));
        const b = Math.floor(255 * (1 - intensity));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Create biomarker network visualization
     */
    createBiomarkerNetwork(containerId, biomarkers) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const networkHtml = `
            <div class="biomarker-network">
                <h4>Biomarker Interaction Network</h4>
                <div class="network-container">
                    <svg width="400" height="300" class="network-svg">
                        ${this.generateNetworkNodes(biomarkers)}
                        ${this.generateNetworkEdges(biomarkers)}
                    </svg>
                </div>
                <div class="network-legend">
                    <div class="legend-item">
                        <div class="legend-color genomic"></div>
                        <span>Genomic</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color transcriptomic"></div>
                        <span>Transcriptomic</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color proteomic"></div>
                        <span>Proteomic</span>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = networkHtml;
    }

    /**
     * Generate network nodes
     */
    generateNetworkNodes(biomarkers) {
        const nodes = [
            { id: 'drug', x: 200, y: 150, type: 'drug', label: 'Drug Response' },
            { id: 'egfr', x: 100, y: 80, type: 'genomic', label: 'EGFR' },
            { id: 'gene_sig', x: 300, y: 80, type: 'transcriptomic', label: 'Gene Signature' },
            { id: 'protein', x: 150, y: 220, type: 'proteomic', label: 'Protein Marker' },
            { id: 'pathway', x: 250, y: 220, type: 'transcriptomic', label: 'Pathway' }
        ];

        return nodes.map(node => {
            const color = this.getFeatureColor(node.type);
            const radius = node.type === 'drug' ? 20 : 15;
            
            return `
                <g class="network-node">
                    <circle cx="${node.x}" cy="${node.y}" r="${radius}" 
                            fill="${color}" stroke="#fff" stroke-width="2"/>
                    <text x="${node.x}" y="${node.y + radius + 15}" 
                          text-anchor="middle" font-size="12" fill="#374151">
                        ${node.label}
                    </text>
                </g>
            `;
        }).join('');
    }

    /**
     * Generate network edges
     */
    generateNetworkEdges(biomarkers) {
        const edges = [
            { from: [100, 80], to: [200, 150], strength: 0.8 },
            { from: [300, 80], to: [200, 150], strength: 0.6 },
            { from: [150, 220], to: [200, 150], strength: 0.4 },
            { from: [250, 220], to: [200, 150], strength: 0.5 }
        ];

        return edges.map(edge => {
            const opacity = edge.strength;
            const strokeWidth = edge.strength * 3 + 1;
            
            return `
                <line x1="${edge.from[0]}" y1="${edge.from[1]}" 
                      x2="${edge.to[0]}" y2="${edge.to[1]}"
                      stroke="#6b7280" stroke-width="${strokeWidth}" 
                      opacity="${opacity}"/>
            `;
        }).join('');
    }
}

// Global visualization manager instance
const visualizationManager = new VisualizationManager();

// Export for use in other modules
window.VisualizationManager = VisualizationManager;
window.visualizationManager = visualizationManager;

// Add CSS styles for visualizations
const visualizationStyles = document.createElement('style');
visualizationStyles.textContent = `
    /* SHAP Plot Styles */
    .shap-plot {
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
        margin: 20px 0;
    }

    .feature-bars {
        margin: 20px 0;
    }

    .feature-bar {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 8px;
        background: #fff;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .feature-name {
        min-width: 150px;
        font-weight: 500;
    }

    .bar-container {
        flex: 1;
        margin: 0 10px;
    }

    .bar {
        height: 20px;
        border-radius: 10px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 8px;
    }

    .bar-value {
        color: white;
        font-size: 12px;
        font-weight: bold;
    }

    .feature-type {
        min-width: 100px;
        font-size: 12px;
        color: #6b7280;
        text-transform: capitalize;
    }

    /* Attention Heatmap Styles */
    .attention-heatmap {
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
        margin: 20px 0;
    }

    .heatmap-grid {
        display: grid;
        gap: 1px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
        margin: 20px 0;
    }

    .heatmap-header {
        display: grid;
        grid-template-columns: 100px repeat(3, 1fr);
    }

    .heatmap-row {
        display: grid;
        grid-template-columns: 100px repeat(3, 1fr);
    }

    .header-cell, .row-header {
        background: #374151;
        color: white;
        padding: 10px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .heatmap-cell {
        background: #f3f4f6;
        height: 40px;
        cursor: pointer;
        transition: opacity 0.2s;
    }

    .heatmap-cell:hover {
        opacity: 0.8;
    }

    .heatmap-legend {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin: 10px 0;
    }

    .legend-gradient {
        width: 100px;
        height: 20px;
        background: linear-gradient(to right, rgb(255, 255, 255), rgb(255, 0, 0));
        border-radius: 10px;
    }

    /* Feature Importance Chart Styles */
    .feature-importance-chart {
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
        margin: 20px 0;
    }

    .chart-bars {
        margin: 20px 0;
    }

    .chart-bar {
        margin: 15px 0;
    }

    .bar-info {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }

    .bar-label {
        font-weight: 500;
    }

    .bar-category {
        font-size: 12px;
        color: #6b7280;
        text-transform: capitalize;
    }

    .bar-track {
        background: #e5e7eb;
        border-radius: 10px;
        height: 20px;
        position: relative;
    }

    .bar-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 8px;
        min-width: 60px;
    }

    .bar-fill .bar-value {
        color: white;
        font-size: 12px;
        font-weight: bold;
    }

    /* Network Visualization Styles */
    .biomarker-network {
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
        margin: 20px 0;
    }

    .network-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    .network-svg {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        background: white;
    }

    .network-legend {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 10px 0;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 50%;
    }

    .legend-color.genomic { background-color: #3b82f6; }
    .legend-color.transcriptomic { background-color: #10b981; }
    .legend-color.proteomic { background-color: #f59e0b; }

    /* General Styles */
    .plot-description {
        font-size: 14px;
        color: #6b7280;
        text-align: center;
        margin-top: 15px;
        font-style: italic;
    }
`;

document.head.appendChild(visualizationStyles);
