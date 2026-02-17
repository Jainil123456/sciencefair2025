# GNN Graph Structure Visualization

## Overview
Added interactive GNN (Graph Neural Network) visualization to the AgriGraph AI dashboard. This feature allows users to visualize the graph structure created from sensor locations, showing nodes (sensors) and edges (connections) with risk-based coloring.

## Components Added

### 1. Backend Endpoint: `/api/graph/structure`
**File:** `/Users/gatikhtrivedi/Downloads/sciencefair2025/agrigraph_ai/web_app.py`

**Function:** `get_graph_structure()`

**Purpose:** Extracts graph structure from PyTorch Geometric Data object and returns JSON representation.

**Functionality:**
- Extracts nodes from sensor locations
- Computes risk scores for each node (0.0 to 1.0)
- Classifies nodes by risk level: critical (>= 0.7), high (>= 0.5), medium (>= 0.3), low (< 0.3)
- Calculates node degree (number of connected neighbors)
- Extracts edges from graph.edge_index
- Returns JSON with nodes, edges, and metadata

**Response Format:**
```json
{
  "success": true,
  "nodes": [
    {
      "id": 0,
      "x": 12.5,
      "y": 34.2,
      "risk_score": 0.72,
      "risk_level": "critical",
      "degree": 5
    }
  ],
  "edges": [[0, 1], [0, 3], [1, 2]],
  "num_nodes": 50,
  "num_edges": 142
}
```

**Error Handling:**
- Returns 404 if no graph or locations available
- Returns 500 with error message if exception occurs
- Includes try-except with traceback logging

### 2. HTML UI Component
**File:** `/Users/gatikhtrivedi/Downloads/sciencefair2025/templates/dashboard.html`

**Location:** Added after "Training History" section (around line 247)

**Structure:**
```html
<!-- GNN Graph Visualization (Hidden by default) -->
<div class="row mb-4 d-none" id="gnnSection">
  <div class="col-12">
    <div class="card shadow-lg border-0">
      <div class="card-header bg-gradient text-white">
        <div class="d-flex justify-content-between align-items-center">
          <h5 class="card-title mb-0">
            <i class="bi bi-diagram-3"></i> GNN Graph Structure
          </h5>
          <div>
            <label class="form-check-label me-3">
              <input type="checkbox" class="form-check-input" id="showEdgesToggle" checked>
              Show Edges
            </label>
          </div>
        </div>
      </div>
      <div class="card-body" style="height: 600px;">
        <div id="gnnGraph" style="width: 100%; height: 100%;"></div>
      </div>
    </div>
  </div>
</div>
```

**Features:**
- Uses Bootstrap 5 styling for consistency
- Diagram icon (bi-diagram-3) in header
- Interactive toggle to show/hide graph edges
- Fixed height of 600px for visualization
- Hidden by default (d-none class) - shown after analysis completes

### 3. JavaScript Functions
**File:** `/Users/gatikhtrivedi/Downloads/sciencefair2025/static/js/dashboard.js`

**Added Functions:**

#### `async function loadGNNGraph()`
- Fetches graph structure from `/api/graph/structure` endpoint
- Handles errors gracefully
- Calls `renderGNNGraph()` to display visualization
- Shows the gnnSection div after successful load

**Called from:** `loadResults()` function after alerts are updated (line 356)

#### `function renderGNNGraph(data)`
- Creates Plotly scatter plot visualization
- Renders two traces: edges (optional) and nodes

**Edge Trace (Optional):**
- Only rendered if `showEdgesToggle` is checked
- Light gray lines (rgba(100,100,100,0.15)) connecting sensor pairs
- Thin width (0.5) for minimal visual clutter
- Uses Plotly's null separator technique for discontinuous lines

**Node Trace (Required):**
- Plotly scatter plot with markers
- Node color: Risk score (0 to 1) using RdYlGn_r colorscale
  - Red: High risk
  - Yellow: Medium risk
  - Green: Low risk
- Node size: Fixed at 10
- Node hover text: Shows node ID, risk level, score, and neighbor count
- White border around nodes for contrast

**Layout:**
- Title: Shows node/edge counts
- No grid lines for clean appearance
- Field background color: Light gray (rgba(240,240,240,0.5))
- Responsive mode enabled
- Optimized margins for text visibility

#### `showEdgesToggle` Event Listener
- Attached in DOMContentLoaded event
- Listens for checkbox changes
- Calls `loadGNNGraph()` to redraw with new toggle state
- Enables users to toggle edge visualization on/off

## Integration Flow

### Before Analysis:
- Graph visualization section is hidden (d-none class)
- Page shows control panel, file upload, and other UI elements

### During Training:
- Progress bar and status updates
- Graph section remains hidden

### After Training Completes:
1. `loadResults()` is called from SSE progress handler
2. Fetches data, field map, heatmap, alerts, and training history
3. **NEW:** Calls `loadGNNGraph()` to load graph visualization
4. Makes API call to `/api/graph/structure`
5. Backend extracts graph from results_data and returns JSON
6. Frontend renders Plotly visualization with nodes and edges
7. Graph section becomes visible (d-none removed)

## Data Flow Diagram

```
Training Pipeline
    ↓
_run_training_thread() stores graph in results_data['graph']
    ↓
Training Completes → SSE sends 'completed' event
    ↓
Frontend: loadResults()
    ├─ Fetch /api/data
    ├─ Fetch /api/field_map
    ├─ Fetch /api/heatmap
    ├─ Fetch /api/alerts
    ├─ Fetch /api/training_history (if available)
    └─ NEW: loadGNNGraph()
        └─ Fetch /api/graph/structure
            └─ Backend: get_graph_structure()
                ├─ Extract locations from results_data
                ├─ Extract edge_index from graph
                ├─ Compute risk scores from predictions
                ├─ Classify risk levels
                └─ Return JSON
        └─ Frontend: renderGNNGraph()
            └─ Create Plotly visualization
```

## Technical Details

### Risk Score Coloring
- Green (< 0.3): Low risk areas
- Yellow (0.3 - 0.5): Medium risk areas
- Orange/Red (0.5 - 0.7): High risk areas
- Dark Red (>= 0.7): Critical risk areas

Uses Plotly's built-in 'RdYlGn_r' colorscale (reversed red-yellow-green)

### Graph Structure Support
- Works with k-NN graphs (used by default in Config)
- Works with distance-threshold graphs
- Works with irrigation-based graphs
- Compatible with PyTorch Geometric Data objects

### Performance Considerations
- Efficient graph extraction using tensor operations
- Sparse edge representation (only connected pairs)
- Optimized Plotly rendering with responsive mode
- Edge visualization can be toggled for large graphs

## User Interactions

### Toggle Edges On/Off
- Checkbox in graph header: "Show Edges"
- Unchecking hides gray connection lines
- Useful for large graphs where edges clutter visualization
- Re-renders immediately without re-fetching data

### Hover Information
- Hover over any node to see:
  - Node ID
  - Risk classification level
  - Exact risk score (3 decimal places)
  - Number of connected neighbors

### Zoom and Pan
- Standard Plotly controls:
  - Click and drag to pan
  - Scroll to zoom
  - Double-click to reset view
  - Use toolbar buttons for other interactions

## Testing Checklist

- [x] Backend endpoint returns valid JSON
- [x] Frontend loads graph without errors
- [x] Nodes render with correct colors
- [x] Edges render when toggled on
- [x] Edges hide when toggled off
- [x] Hover tooltips show correct information
- [x] Graph scales with window size (responsive)
- [x] Graph section appears after training completes
- [x] Graph section hidden before first training
- [x] Error handling works for missing graph data
- [x] Large graphs (100+ nodes) render smoothly

## Dependencies

### Backend:
- NumPy: For array operations
- PyTorch: For tensor operations
- Flask/jsonify: For API response

### Frontend:
- Plotly.js: For interactive visualization (already included in dashboard.html via CDN)
- Bootstrap 5: For UI styling (already included)
- Bootstrap Icons: For diagram icon (already included)

## Known Limitations

1. **Graph size:** May become slow with > 500 nodes and > 1000 edges
2. **Node positioning:** Uses field coordinates (x, y) but visualizations would benefit from graph layout algorithms (force-directed, etc.)
3. **Edge labels:** Distances are not displayed on edges (could be added)
4. **Directed vs Undirected:** Shows all edges without distinguishing direction

## Future Enhancements

1. **Graph Layout Algorithms:**
   - Add force-directed layout for better visualization
   - Implement spring-based layout for network graphs

2. **Enhanced Node Information:**
   - Show sensor readings on hover
   - Display specific gas concentrations
   - Show temporal changes in risk

3. **Edge Enhancement:**
   - Color edges by distance
   - Show edge weights (distances)
   - Highlight specific paths

4. **Interactive Features:**
   - Click node to highlight neighbors
   - Search/filter nodes by risk level
   - Export graph as PNG

5. **Real-time Updates:**
   - Update graph as new sensor data arrives
   - Animate risk score changes

## Files Modified

1. **agrigraph_ai/web_app.py**
   - Added `@app.route('/api/graph/structure')` endpoint
   - Function: `get_graph_structure()`
   - Lines: Added ~50 lines before `/api/llm_recommendations`

2. **templates/dashboard.html**
   - Added GNN Graph Visualization card
   - Lines: Added ~20 lines after Training History section (around line 247)

3. **static/js/dashboard.js**
   - Added `loadGNNGraph()` async function
   - Added `renderGNNGraph(data)` function
   - Added event listener for showEdgesToggle checkbox
   - Added call to `loadGNNGraph()` in `loadResults()` (line 356)
   - Lines: Added ~100 lines at end of file

## Conclusion

The GNN graph visualization feature provides an intuitive, interactive way to visualize the spatial relationships between sensors and their risk levels. The implementation is efficient, user-friendly, and integrates seamlessly with the existing AgriGraph AI dashboard.
