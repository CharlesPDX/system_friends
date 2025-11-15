# Dual-System AI Chatbot

A FastAPI-based conversational AI system that implements a dual-process cognitive architecture inspired by human thinking, featuring System 1 (fast, intuitive) and System 2 (slow, deliberative) processing modes with metacognitive monitoring.

## Overview

This application demonstrates a sophisticated AI architecture where:

- **System 1** provides quick, intuitive responses to user queries
- **Metacognitive State Vector (MSV)** monitors response quality across multiple dimensions
- **System 2** engages when the MSV indicates deeper reasoning is needed
- **Interactive visualization** displays the decision-making process using radar charts and node graphs

## Features

- 🧠 **Dual-Process Architecture**: Automatic engagement of System 2 based on metacognitive assessment
- 📊 **Real-time Visualization**: Interactive Bokeh charts showing metacognitive state vectors
- 💬 **Chat Interface**: HTMX-powered responsive chat UI
- 📈 **Multi-dimensional Analysis**: Tracks emotional response, correctness, experiential matching, conflict detection, and problem importance
- 🗄️ **Session Recording**: SQLite-based interaction history, including the prompts and weight parameters of the overall system
- 🎯 **Node Graph Visualization**: Shows System 2's reasoning process across multiple specialized nodes
- ⚙️ **Configurable Weights**: Adjustable parameters for MSV calculations and prompts

## Architecture Components

### Metacognitive State Vector (MSV)

The MSV evaluates responses across five key dimensions:

1. **Emotional Response** - Affective quality of the response
2. **Correctness** - Accuracy and reliability assessment
3. **Experiential Matching** - Alignment with prior knowledge
4. **Conflict Information** - Detection of contradictions or uncertainties
5. **Problem Importance** - Significance assessment

Each dimension is calculated with weighted sub-components and visualized using radar charts.

### System 2 Processing

When the MSV activation threshold is exceeded, System 2 engages for deeper analysis through:

- Specialized reasoning nodes
- Multi-step deliberative processing, where each node/step can take on different roles based on the MSV
- Enhanced response generation
- Node-level decision tracking

## Installation

```bash
# Install ollama from https://ollama.com
# Install the llama3.2 model
ollama pull llama3.2

# Clone the repository
git clone https://github.com/CharlesPDX/system_friends.git 
cd system_friends

# Setup venv
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install text blob corpora
python -m textblob.download_corpora

# Run System One
python3 app.py --system-two-url http://127.0.0.1:8001

# Run System Two (in a different terminal tab, rember to activate the same virtual environment)
python3 app.py --system-two

# Open a browser tab to http://localhost:8000 - you should get the demo page

# Enter some text, and click Send

# Wait, and once the response is returned, you can click on the gray box to get the report out (will also take a minute)

# You can use beekeeper studio (or whatever you like for SQLite files) and get the interactions from the system_friends/data directory, opening the appropriate .sqlite3 file

# You can see the parameters the interactions are running with from the parameters table, and the interactions (System One and Two MSVs and responses along with the user input) from the interactions table.
```

### Required Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `httpx` - HTTP client for System 2 requests
- `bokeh` - Interactive visualization
- `jinja2` - Template engine

## Usage

```bash
# Terminal 1 - System 1 (primary interface)
python main.py --system-two --system-two-url http://localhost:8001

# Terminal 2 - System 2 (reasoning engine)
python main.py --system-two
```

## Project Structure

```
├── main.py                          # FastAPI application and routing
├── system_one_model.py              # Quick response generation
├── system_two_model.py              # Deliberative reasoning
├── metacognitive.py                 # MSV calculation logic
├── prompts.py                       # System prompts configuration
├── history.py                       # Database interaction tracking
├── app_graph.py                     # System 2 node graph visualization
├── templates/
│   ├── admin_panel.html             # Admin interface to adjust weights & prompts
│   ├── index.html                   # Main chat interface
│   └── msv_visualizer.html          # Chart visualization template
├── static/                          # Static assets (CSS, JS)
└── data/                            # SQLite session databases
```

## API Endpoints

### User Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Submit user message and receive response
- `GET /get_chart?id={id}` - Retrieve MSV visualizations for a response
- `GET /node/{node_id}` - Get details for System 2 reasoning nodes
- `POST /reset` - Reset conversation with a new configuration

### System Endpoints

- `POST /system1` - Programmatic access to System 1
- `POST /system2` - Programmatic access to System 2

## Configuration

The system supports runtime configuration of:

- **Weights**: Adjustment factors for each MSV component and sub-component
- **Prompts**: System instructions and evaluation criteria

Configuration can be modified through the web interface and is saved with each session.

## Data Storage

Each session creates a SQLite database in the `data/` directory:

- Session ID format: `YYYY-MM-DD_HH_MM_SS_ffffff`
- Stores: user prompts, system responses, MSV values, configuration
- Enables post-hoc analysis and system tuning

## Visualization

The system provides multiple visualization types:

1. **Overall MSV** - Aggregate metacognitive state
2. **Component Vectors** - Individual dimension breakdowns (5 charts)
3. **System 2 Node Graph** - Interactive reasoning process visualization

All charts use radar plot format for multi-dimensional data representation.

## Development

### Adding Custom MSV Components

1. Define new sub-components in `metacognitive.py`
2. Update weight initialization in `get_weights()`
3. Add visualization in `_generate_chart()`

### Extending System 2

1. Implement new node types in `system_two_model.py`
2. Update graph generation in `app_graph.py`
3. Add node detail templates as needed

## Command-Line Arguments

- `--system-two`: Run as System 2 instance (listening mode)
- `--system-two-url`: URL of System 2 instance (when running System 1)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Citation

If you use this system in research, please cite:

```
[Add citation information]
```

## Acknowledgments

This implementation is inspired by dual-process theories in cognitive science and metacognitive monitoring research.
