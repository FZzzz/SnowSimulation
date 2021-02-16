# SnowSimulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of my graduation thesis - Position Based Snow Simulation with Phase Change

## Introduction
The simulation of snow is widely used in various places. Most snow are naturally accompanying with abundant water resources. When the environmental temperature is near to the melting point, the existence of water will make the snow have drastic phase changes. 
This leads to different elastoplastic characteristics of snow and will further affect the formulation of snow shape. 

In our research, we introduce a method based on Position Based Dynamics by using two different discretization methods, Discrete Element Method (DEM) and Smoothed Particle Hydrodynamics (SPH) to simulate the interaction of surrounding water and ice crystals in snow. 
By introducing stretch constraints to DEM particles which performs as the interlinks of snow particles, we successfully simulate the deformation effect. 
In addition, we address the problem of over-connected interlinks by controlling the number of connections with considering the homogeneous freezing effects of ice crystals in snow. 

Our method is a fully GPU-based algorithm which addresses the complex implementation of phase-change problems. 
As a result, our method is able to perform various characteristics of snow including deformation, phase change, and the rigid-fluid interactions. 
Despite of the limitations caused by our choice of constraint and the uniform size of particles, our position-based solver is a stable and controllable solution for simulating the complex behaviors of snow-water interactions.

## Build Dependency
- CUDA 11.0+
- OpenGL 4.0+
- [GLFW](https://github.com/glfw/glfw)
- [SOIL](https://github.com/kbranigan/Simple-OpenGL-Image-Library)
- [Partio](https://github.com/wdas/partio)
  - for .bgeo export
- imgui (has been included in the `\include` folder)

## Build Instruction
Open `SnowSimulation.sln` and build it with Visual Studio 2019

(If there's lib missing while building, please refer to the prebuilt libs in `\libs`)

## Result
![](https://i.imgur.com/8KKHWOq.png)
![](https://i.imgur.com/zRV2xnW.png)
![](https://i.imgur.com/3ELWO6c.png)
![](https://i.imgur.com/DZ3hbnv.png)

| Parameter | Value |
| -------- | -------- |
|Snow temperature | −10 (C◦) |
|Water temperature | 100 (C◦) |
|Snow heat conductivity | 2500 (W/mK)|
|Water heat conductivity | 600 (W/mK)|
|Number of water particles | 5888|
|Number of snow particles | 147928|


|  Per frame time |          |
| -------- | -------- |
| Min  | 41.025 (ms)|
| Max  | 73.729 (ms) |
| Avg. | 56.809 (ms) |


**For more information, please reference to our paper (should be published in near future by the school)**

## License
Copyright 2021 Chen Yi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
