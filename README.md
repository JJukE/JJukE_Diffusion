Diffusion bases for other various models

# Contents

```bash
.
|-- __init__.py
|-- common.py
|-- diffusion
|   |-- __init__.py
|   |-- common.py
|   |-- ddim.py
|   |-- ddpm.py
|   |-- diffusion_base.py
|   |-- karras.py
|   `-- ldm.py
`-- unet
    |-- __init__.py
    |-- base_modules.py
    |-- ldm_unet.py
    |-- transformer.py
    |-- unet_base.py
    |-- unet_cond.py
    `-- unet_modules.py
```

# To-do List

- [ ] Test 1D U-Net
- [ ] Implement Conditional U-Net (need to study classifier free guidance)
- [ ] Trouble shooting of LDM U-Net (OOM)
- [ ] Implement 3D U-Net