# Hyperblock Classifier

Two Hyperblock classifiers are available:
- [VisCanvas2.0](https://github.com/CWU-VKD-LAB/VisCanvas2.0): Sequentially builds HBs starting from the first point in the file, then removes cases in the first HB.  
- [DV2.0](https://github.com/CWU-VKD-LAB/DV2.0): Collects intervals on the attributes.  

Both are implemented as standalone Python modules in [`../../software/BAP rebuild`](../../software/BAP%20rebuild/) with `fit` and `predict` compatible with BAP:

| Module | BAP name | Description |
|:-------|:---------|:------------|
| `hb_viscanvas.py` | `hb_vis` or `hb_viscanvas` | VisCanvas-style: builds pure HBs sequentially, removes covered cases, repeats |
| `hb_dv.py` | `hb_dv` | DV-style: collects attribute intervals, forms axis-aligned hyperblocks per class |

**Usage:**
```bash
python bap.py -c config.toml --classifier hb_vis
python bap.py -c config.toml --classifier hb_dv
```

Or in TOML: `classifier = "hb_vis"` or `classifier = "hb_dv"`.
