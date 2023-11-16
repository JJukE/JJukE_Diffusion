Diffusion bases for other various models

# Contents

```bash
.
|-- README.md
|-- __init__.py
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
|   |-- unet_base.py
|   `-- unet_modules.py
`-- unet_cond
    |-- __init__.py
    |-- attention.py
    |-- fp16_util.py
    |-- transformer.py
    |-- unet.py
    `-- unet_modules.py
```

# To-do List

- [ ] Test 1D U-Net
- [x] Implement Conditional U-Net
- [x] Test LDM U-Net
- [ ] Enjoy it!