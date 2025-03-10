# Advanced Fractal Geometry Generator for Blender

## Overview

The Advanced Fractal Geometry Generator is a powerful Blender addon that creates complex, fractal-based geometry modifications. It transforms selected faces using mathematical fractal patterns to create intricate, organic structures with minimal effort.

![Fractal Generator](https://via.placeholder.com/800x400.png?text=Fractal+Generator+Example)

## Features

- **Multiple Fractal Types**: Choose from classic 2D fractals (Mandelbrot, Julia) and 3D fractals (Cubic and Quintic Mandelbulbs)
- **Detailed Control**: Fine-tune your fractal patterns with controls for scale, complexity, and iterations
- **Stepping Pattern System**: Create complex geometry using the extrude → inset → extrude pattern
- **Face Selection**: Work on all faces or only selected faces
- **Batch Processing**: Process large meshes efficiently with controlled batch sizes
- **Performance Safeguards**: Prevent system slowdowns with intelligent limits and warnings

## Installation

1. Download the latest release from the GitHub repository
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the downloaded .zip file
4. Enable the addon by checking the box next to "Mesh: Advanced Fractal Geometry Generator"

## Usage

### Basic Operation

1. Select a mesh object in Blender
2. Enter Edit Mode and select faces (optional - if "Selected Faces Only" is enabled)
3. Open the Fractal tab in the sidebar (View3D > Sidebar > Fractal)
4. Adjust settings as desired
5. Click "Generate Fractal"

### Settings

#### Basic Settings

- **Fractal Type**: Choose the mathematical pattern to apply
  - Mandelbrot: Classic 2D Mandelbrot set
  - Julia: Classic 2D Julia set
  - Quintic Mandelbulb: 3D fractal with power=5
  - Cubic Mandelbulb: 3D fractal with power=3
- **Scale**: Controls the scale of the fractal pattern
- **Complexity**: Affects extrusion height and pattern intricacy
- **Selected Faces Only**: When enabled, only works on selected faces
- **Face Limit**: Maximum number of faces to process for performance

#### Stepping Pattern Settings

##### First Extrusion
- **First Extrusion Amount**: Base amount for initial extrusion
- **Extrude Along Normals**: When enabled, extrudes along face normals (otherwise uses global Z)

##### Inset Controls
- **Inset Amount**: Controls how much faces are inset
- **Inset Depth**: Depth of inset (positive for outward, negative for inward)
- **Relative Inset**: Scales inset by face size for more uniform results
- **Edge Inset Only**: When enabled, only edges are affected by inset

##### Second Extrusion
- **Second Extrusion Factor**: Factor relative to first extrusion
- **Use Individual Normals**: When enabled, uses each face's normal for more variation

#### Advanced Settings
- **Iterations**: Number of fractal iterations (higher values create more detail but slower)
- **Random Seed**: Seed for random generation (allows reproducible results)
- **Smooth Shading**: Applies smooth shading to the result

#### Safety Settings
- **Batch Processing**: Processes faces in batches for better UI responsiveness
- **Batch Size**: Number of faces to process in each batch

## Tips and Tricks

- Start with a simple shape and low face count to see how the settings affect the result
- Use the "Randomize Seed" button to quickly explore different variations
- If performance is slow, try reducing the number of iterations or face limit
- For the most intricate results, try the Mandelbulb fractals with high iteration counts
- The "Complexity" setting dramatically affects the resulting geometry - start with lower values
- Use batch processing for large meshes to keep Blender responsive during generation

## Performance Considerations

- High iteration counts (>100) may cause slow processing
- Large face counts can significantly impact performance
- The 3D Mandelbulb fractals are more computationally intensive than 2D fractals
- If Blender becomes unresponsive, you can always cancel the operation

## Troubleshooting

- **No faces are being modified**: Ensure you have faces selected if "Selected Faces Only" is enabled
- **Blender crashes**: Try reducing face limit, batch size, or iterations
- **Strange geometry artifacts**: Check for very small inset or extrusion values
- **Processing seems stuck**: For large operations, check the progress bar or use the Cancel button

## Compatible Blender Versions

This addon is designed for Blender 4.3.0 and newer.

## Development Status

⚠️ **WORK IN PROGRESS** - This addon is currently under active development and not ready for production use.

## License

Apache License 2.0

## Credits

Created by Gero Doll aka Limbicnation
