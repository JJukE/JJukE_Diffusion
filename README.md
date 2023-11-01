Diffusion bases for other various models

# Contents

```bash
.
|-- README.md
|-- __init__.py
|-- common.py
|-- diffusion
|   |-- __init__.py
|   |-- common.py
|   |-- ddim.py
|   |-- ddpm.py
|   |-- diffusion_base.py
|   |-- gaussian_diffusion.py
|   |-- karras.py
|   `-- ldm.py
|-- unet
|   |-- __init__.py
|   |-- base_modules.py
|   |-- ldm_unet.py
|   |-- transformer.py
|   |-- unet_base.py
|   |-- unet_cond.py
|   `-- unet_modules.py
`-- unet_ldm
    |-- attention.py
    |-- unet.py
    `-- unet_modules.py
```

# To-do List

- [ ] Test 1D U-Net
- [ ] Implement Conditional U-Net
- [ ] Test LDM U-Net
- [ ] Implement 3D U-Net