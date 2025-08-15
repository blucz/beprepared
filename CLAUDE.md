# BePrep - Image Dataset Preparation Tool

## Project Overview
BePrep is a Python-based tool for preparing high-quality image datasets for diffusion model fine-tuning. It provides a powerful workflow DSL for processing images with both automated and human-in-the-loop operations.

## Architecture

### Core Components

1. **Node System** (`beprepared/node.py`)
   - Base class for all processing nodes
   - Uses metaclass to enable DSL patterns like `Node1 >> Node2`
   - Each node processes datasets and returns new datasets
   - Supports chaining via `>>` and `<<` operators

2. **Dataset** (`beprepared/dataset.py`)
   - Container for images being processed
   - Supports copying for non-destructive operations

3. **Image** (`beprepared/image.py`)
   - Represents individual images with properties
   - Uses PropertyBag pattern for flexible attributes
   - Allowed formats: JPEG, PNG, WEBP, GIF, TIFF, BMP

4. **Properties System** (`beprepared/properties.py`)
   - `PropertyBag`: Base class for objects holding properties
   - `CachedProperty`: Properties cached in SQLite database
   - `ConstProperty`: Immutable properties
   - `ComputedProperty`: Dynamically computed properties

5. **Workspace** (`beprepared/workspace.py`)
   - Manages global state and database
   - SQLite database for caching operations
   - Thread-local database connections
   - Object storage for image data

6. **Web Interface** (`beprepared/web.py`, `beprepared/web/`)
   - FastAPI-based web server for human-in-the-loop tasks
   - Vue.js frontend for filtering and tagging interfaces
   - Real-time progress updates via WebSockets

## Node Categories

### Data Loading & Saving
- `Load`: Load images from directory
- `Save`: Save processed images

### Image Processing
- `ConvertFormat`: Convert image formats
- `Upscale`/`Downscale`: Resize images
- `Anonymize`: Blur faces in images
- `EdgeWatermarkRemoval`: Remove watermarks

### Filtering & Selection
- `FilterBySize`: Filter by image dimensions
- `FilterByAspectRatio`: Filter by aspect ratio
- `HumanFilter`: Manual human filtering via web UI
- `SmartHumanFilter`: Intelligent human filtering

### Captioning (Multiple VLM providers)
- `JoyCaptionAlphaOne`/`JoyCaptionAlphaTwo`: JoyCaption models
- `GPT4oCaption`: OpenAI GPT-4 Vision
- `GeminiCaption`: Google Gemini
- `LlamaCaption`: Meta Llama
- `QwenVLCaption`: Qwen Vision-Language
- `XGenMMCaption`: xGen multimodal
- `Florence2Caption`: Microsoft Florence2
- `MolmoCaption`: Molmo captioning

### Analysis & Scoring
- `NudeNet`: NSFW content detection
- `AestheticScore`: Aesthetic quality scoring
- `ClipEmbed`: CLIP embeddings

### Tagging & Metadata
- `HumanTag`: Manual tagging via web UI
- `AddTags`/`RemoveTags`/`RewriteTags`: Tag manipulation
- `LLMCaptionTransform`: Transform captions using LLMs
- `LLMCaptionVariations`: Generate caption variations

### Deduplication
- `ExactDedupe`: Exact duplicate removal
- `FuzzyDedupe`: CLIP-based fuzzy deduplication

### Utility Nodes
- `Info`: Print dataset information
- `Concat`: Concatenate multiple datasets
- `Take`: Take first N images
- `Sorted`: Sort images by property
- `Shuffle`: Randomize order
- `Map`/`Apply`/`Filter`: Functional operations

## Creating New Nodes

### Basic Node Template

```python
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty

class MyNode(Node):
    '''Node description for documentation'''
    
    def __init__(self, param1: str, param2: int = 10):
        '''Initialize the node
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2 (default: 10)
        '''
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def eval(self, dataset: Dataset) -> Dataset:
        '''Process the dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Processed dataset
        '''
        # Process each image
        for image in dataset.images:
            # Access existing properties
            width = image.width.value
            height = image.height.value
            
            # Add new cached property
            result_prop = CachedProperty('mynode_result', image)
            if not result_prop.has_value:
                # Compute and cache result
                result = self.process_image(image)
                result_prop.value = result
            
            # Add property to image
            image.mynode_result = result_prop
        
        return dataset
    
    def process_image(self, image):
        # Your processing logic here
        pass
```

### Key Patterns

1. **Property Caching**: Use `CachedProperty` to avoid recomputing expensive operations
2. **Non-destructive**: Always return a new dataset or copy
3. **Logging**: Use `self.log` for logging (automatically connected to web UI)
4. **Progress**: Use `tqdm` from `beprepared.nodes.utils` for progress bars
5. **Web Integration**: For human-in-the-loop, see `HumanFilter`/`HumanTag` examples

### Parallel Processing Pattern

For GPU-intensive operations, use the `ParallelWorker` pattern:

```python
from beprepared.nodes.parallelworker import ParallelWorker

class MyGPUNode(ParallelWorker):
    def __init__(self):
        super().__init__(num_workers=2)  # Number of GPU workers
    
    def load_models(self):
        # Load models once per worker
        import torch
        self.model = load_my_model()
    
    def process_image(self, image):
        # Process single image on GPU
        result = self.model(image)
        return result
```

## Workflow Examples

### Basic Workflow
```python
from beprepared import *

(
    Load("input_images")
    >> FilterBySize(min_edge=512)
    >> JoyCaptionAlphaOne
    >> Save("output")
)
```

### Complex Workflow with Human Tasks
```python
(
    Load("raw_images")
    >> FilterBySize(min_edge=512)
    >> HumanFilter                    # Web UI for filtering
    >> Anonymize                      # Blur faces
    >> JoyCaptionAlphaOne             # Auto-caption
    >> HumanTag(tags=["style1", "style2"])  # Manual tagging
    >> LLMCaptionTransform(           # Enhance captions
        system_prompt="Improve this caption",
        user_prompt="Caption: {caption}"
    )
    >> Save("final_dataset")
)
```

## Testing New Nodes

1. Create test script in project root:
```python
from beprepared import *
from beprepared.nodes.mynode import MyNode

(
    Load("test_images")
    >> MyNode(param1="test")
    >> Info
    >> Save("test_output")
)
```

2. Run with CLI:
```bash
beprepared run test_script.py
```

## Database Schema

The workspace uses SQLite with:
- `property_cache`: Cached properties (key, domain, value, timestamp)
- `objects`: Stored image data (objectid, data)
- `migrations`: Schema version tracking

## Important Conventions

1. **Immutable Images**: Once loaded, images are considered immutable
2. **Caching**: All expensive operations should be cached
3. **Non-destructive**: Never modify original images
4. **Progress Feedback**: Use tqdm for long operations
5. **Web Integration**: Human tasks automatically launch web UI

## CLI Commands

- `beprepared run <workflow.py>`: Execute workflow file
- `beprepared exec "<pipeline>"`: Quick one-liner execution
- `beprepared db list [pattern]`: List cached properties
- `beprepared db clear [pattern]`: Clear cached data

## Dependencies

Key libraries:
- PyTorch & torchvision
- FastAPI & uvicorn (web interface)
- Vue.js (frontend)
- Pillow (image processing)
- OpenAI, LiteLLM (LLM integrations)
- CLIP, transformers (ML models)
- SQLite3 (caching)

## Development Tips

1. **Add to __init__.py**: Export new nodes in `beprepared/nodes/__init__.py`
2. **Documentation**: Add docstrings for auto-generated docs
3. **Error Handling**: Use try/except and log errors with `self.log.exception()`
4. **Testing**: Test with small datasets first
5. **GPU Memory**: Be mindful of GPU memory when processing batches

## Project Structure
```
beprepared/
├── __init__.py           # Main exports
├── cli.py                # CLI interface
├── node.py               # Base Node class
├── dataset.py            # Dataset container
├── image.py              # Image class
├── properties.py         # Property system
├── workspace.py          # Global state & DB
├── web.py                # Web server
├── nodes/                # All node implementations
│   ├── __init__.py       # Node exports
│   ├── load.py           # Load node
│   ├── save.py           # Save node
│   ├── humanfilter.py    # Human filtering
│   ├── humantag.py       # Human tagging
│   └── ...               # Other nodes
└── web/                  # Frontend code
    ├── App.vue           # Main app
    ├── HumanFilter.vue   # Filter UI
    └── HumanTag.vue      # Tag UI
```