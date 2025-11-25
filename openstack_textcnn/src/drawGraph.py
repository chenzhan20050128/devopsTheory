from graphviz import Digraph


def draw_textcnn_architecture():
    """Draw TextCNN architecture diagram, consistent with actual model parameters (128-dim embedding, 128 filters)"""
    # Create directed graph with top-to-bottom layout for better line alignment
    dot = Digraph(comment="TextCNN Architecture", format="png")
    # Improved layout: clearer spacing and straight lines
    dot.attr(
        rankdir="TB",  # Top to Bottom for better line alignment
        splines="polyline",  # Use polyline for straighter lines
        nodesep="1.0",
        ranksep="1.2",
        dpi="300",  # High resolution
        bgcolor="white",
        concentrate="false",  # Don't merge edges
    )

    # Global node style: improved font and style
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="white",
        fontname="Arial",
        fontsize="11",
        height="0.6",
        width="1.8",
        margin="0.3",
    )

    # Global edge style - use straight lines
    dot.attr("edge", fontname="Arial", fontsize="9", arrowsize="0.8", 
             style="solid", color="black")

    # ============================================================
    # 1. Input Layer
    # ============================================================
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(
            label="Input Layer",
            style="rounded,filled",
            color="#2E7D32",
            fillcolor="#E8F5E9",
            fontname="Arial Bold",
            fontsize="12",
            labelloc="t",
        )
        c.node(
            "Input_Log",
            "Log Sequence\n(Batch, Seq_Len)",
            fillcolor="#BBDEFB",
            style="rounded,filled,bold",
        )
        c.node(
            "Input_Num",
            "Numeric Features\n(Batch, 5)",
            fillcolor="#C8E6C9",
            style="rounded,filled,bold",
        )

    # ============================================================
    # 2. Embedding Layer - 128 dimensions
    # ============================================================
    dot.node(
        "Embedding",
        "Embedding Layer\n(Batch, 128, Seq_Len)\nVocab → 128-dim",
        fillcolor="#FFF59D",
        style="rounded,filled,bold",
    )
    dot.edge("Input_Log", "Embedding", color="#1976D2", penwidth="2.5")

    # ============================================================
    # 3. Convolution & Pooling Layer - 128 filters
    # ============================================================
    with dot.subgraph(name="cluster_convs") as c:
        c.attr(
            label="Multi-scale Convolution & Max Pooling",
            style="rounded,filled",
            color="#C62828",
            fillcolor="#FFEBEE",
            fontname="Arial Bold",
            fontsize="12",
            labelloc="t",
        )

        # Branch K=3 - 128 filters
        c.node(
            "Conv3",
            "Conv1d (k=3)\n128 Filters\nReLU",
            fillcolor="#FF8A80",
            style="rounded,filled",
        )
        c.node(
            "Pool3",
            "Global MaxPool\nOutput: 128-dim",
            fillcolor="#FF5252",
            style="rounded,filled",
        )

        # Branch K=4 - 128 filters
        c.node(
            "Conv4",
            "Conv1d (k=4)\n128 Filters\nReLU",
            fillcolor="#FF8A80",
            style="rounded,filled",
        )
        c.node(
            "Pool4",
            "Global MaxPool\nOutput: 128-dim",
            fillcolor="#FF5252",
            style="rounded,filled",
        )

        # Branch K=5 - 128 filters
        c.node(
            "Conv5",
            "Conv1d (k=5)\n128 Filters\nReLU",
            fillcolor="#FF8A80",
            style="rounded,filled",
        )
        c.node(
            "Pool5",
            "Global MaxPool\nOutput: 128-dim",
            fillcolor="#FF5252",
            style="rounded,filled",
        )

    # Connect embedding to convolution layers
    dot.edge("Embedding", "Conv3", color="#C62828", penwidth="2.5")
    dot.edge("Embedding", "Conv4", color="#C62828", penwidth="2.5")
    dot.edge("Embedding", "Conv5", color="#C62828", penwidth="2.5")

    # Connect convolution to pooling
    dot.edge("Conv3", "Pool3", color="#D32F2F", penwidth="2")
    dot.edge("Conv4", "Pool4", color="#D32F2F", penwidth="2")
    dot.edge("Conv5", "Pool5", color="#D32F2F", penwidth="2")

    # ============================================================
    # 4. Numeric Feature Processing
    # ============================================================
    with dot.subgraph(name="cluster_numeric") as c:
        c.attr(
            label="Numeric Feature Encoder",
            style="rounded,filled",
            color="#2E7D32",
            fillcolor="#E8F5E9",
            fontname="Arial Bold",
            fontsize="12",
            labelloc="t",
        )
        c.node(
            "Num_Process",
            "Numeric Encoder\nLayerNorm → Linear(5→32)\nReLU → Dropout(0.5)",
            fillcolor="#81C784",
            style="rounded,filled",
        )

    dot.edge("Input_Num", "Num_Process", color="#2E7D32", penwidth="2.5")

    # ============================================================
    # 5. Feature Fusion & Output
    # ============================================================
    dot.node(
        "Concat",
        (
            "Feature Concatenation\nText: 384-dim (128×3)\n"
            "Numeric: 32-dim\nTotal: 416-dim"
        ),
        shape="box3d",
        fillcolor="#B0BEC5",
        style="filled,bold",
    )

    # Aggregate all features to concatenation layer
    dot.edge("Pool3", "Concat", color="#424242", penwidth="2")
    dot.edge("Pool4", "Concat", color="#424242", penwidth="2")
    dot.edge("Pool5", "Concat", color="#424242", penwidth="2")
    dot.edge("Num_Process", "Concat", color="#2E7D32", penwidth="2")

    # Fully connected and output
    dot.node(
        "FC",
        "Fully Connected Layer\nLinear(416 → 128)\nReLU → Dropout(0.5)",
        fillcolor="#BA68C8",
        style="rounded,filled",
    )
    dot.node(
        "Output",
        "Classifier\nLinear(128 → 1)\nSigmoid → Probability",
        shape="ellipse",
        fillcolor="#F06292",
        style="filled,bold",
    )

    dot.edge("Concat", "FC", color="#7B1FA2", penwidth="2.5")
    dot.edge("FC", "Output", color="#C2185B", penwidth="3")

    # ============================================================
    # Render
    # ============================================================
    try:
        from pathlib import Path
        output_dir = Path(__file__).parent
        output_path = dot.render(str(output_dir / "textcnn_architecture"), view=False, cleanup=True)
        print(f"Architecture diagram generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"Generation failed, please ensure Graphviz is installed: {e}")
        print("You can copy the above code to an online Graphviz editor to view.")
        return None


if __name__ == "__main__":
    # Generate architecture diagram
    draw_textcnn_architecture()