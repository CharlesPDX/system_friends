from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, MultiLine, HoverTool, CustomJS, Arrow, VeeHead
from bokeh.embed import components
import networkx as nx

def create_system_two_node_graph():
    """Create the Bokeh graph"""
    # Create a graph with multiple nodes and edges
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (1, 3)
    ])
    
    # Create the Bokeh plot
    plot = figure(
        width=600, 
        height=400, 
        x_range=(-3, 3), 
        y_range=(-3, 3),
        title="Interactive Graph",
        tools="tap,pan,wheel_zoom,reset"
    )
    
    # Create graph renderer
    graph = GraphRenderer()
    
    # Node data
    node_indices = list(G.nodes())
    graph.node_renderer.data_source.data = dict(
        index=node_indices,
        name=[f"Node {i}" for i in node_indices],
        description=[f"This is node {i}" for i in node_indices],
        node_id=node_indices
    )
    
    # Edge data
    edge_start = [edge[0] for edge in G.edges()]
    edge_end = [edge[1] for edge in G.edges()]
    graph.edge_renderer.data_source.data = dict(start=edge_start, end=edge_end)
    
    # Use spring layout for automatic positioning
    graph_layout = nx.spring_layout(G, scale=2, seed=42)
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    
    # Style nodes
    graph.node_renderer.glyph = Circle(radius=0.25, fill_color="lightblue", line_color="navy", line_width=2)
    graph.node_renderer.selection_glyph = Circle(radius=0.25, fill_color="yellow", line_color="navy", line_width=2)
    graph.node_renderer.hover_glyph = Circle(radius=0.3, fill_color="lightgreen", line_color="navy", line_width=2)
    
    # Style edges with arrows
    graph.edge_renderer.glyph = MultiLine(line_color="gray", line_alpha=0.8, line_width=2)
    
    # Add arrows for each edge (aligned to circle edges)
    circle_radius = 0.25
    for start, end in G.edges():
        start_pos = graph_layout[start]
        end_pos = graph_layout[end]
        
        # Calculate direction vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2)**0.5
        
        # Normalize and offset by circle radius
        if distance > 0:
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            x_start = start_pos[0] + dx_norm * circle_radius
            y_start = start_pos[1] + dy_norm * circle_radius
            x_end = end_pos[0] - dx_norm * circle_radius
            y_end = end_pos[1] - dy_norm * circle_radius
            
            arrow = Arrow(
                end=VeeHead(size=15, fill_color="gray"),
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end,
                line_color="gray",
                line_width=2
            )
            plot.add_layout(arrow)
    
    # Add hover tooltip
    hover = HoverTool(
        renderers=[graph.node_renderer],
        tooltips=[
            ("Name", "@name"),
            ("Description", "@description"),
            ("ID", "@node_id")
        ],
        sort_by="distance"  # Options: "distance", "value", or field name like "@node_id"
    )
    plot.add_tools(hover)
    
    # Add tap/click event
    tap_callback = CustomJS(args=dict(source=graph.node_renderer.data_source), code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const nodeId = source.data['node_id'][indices[0]];
            htmx.ajax('GET', '/node/' + nodeId, {
                target: '#node-info',
                swap: 'innerHTML'
            });
        }
    """)
    
    graph.node_renderer.data_source.selected.js_on_change('indices', tap_callback)
    
    # Add graph to plot
    plot.renderers.append(graph)
    
    return plot