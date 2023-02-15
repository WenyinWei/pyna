# pyna

**P**ython package dedicated for research on **DYNA**mic system.

```bash
# Release version
pip install pyna-chaos

# Latest version
git clone git@github.com:WenyinWei/pyna.git
cd pyna
pip install -e .
# git pull for latest update 
```

```python
import pyna
```

Plan to support:
- One dimensional Map / Flow
- Two dimensional Map / Flow
    - Eigenvalue/vector
    - Asymptotic Behaviour Analysis
        - Topology Entropy (How bending are the curves after long-term evolution?)
        - Lyapunov Exponent Spectrum
    - Bifurcation diagram / Bifurcation curve 
    - Stable/Unstable manifold drawing
- Three dimensional Flow Visualization
