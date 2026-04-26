# Capsule SMPL Variants (MimicKit-Compatible)

These variants are generated from:

- `data/assets/smpl/smpl.xml`

using:

- `tools/generate_smpl_capsule_variants.py`

## Why this set works better

Unlike the older files in `data/assets/smpl/human_variants/`, this set keeps the exact same
body/joint/actuator topology as `smpl.xml`.

The older `human_variants/` and `human_variants_mesh/` directories are SMPLSim-style exports
with a larger hand-heavy topology. They are not drop-in replacements for motion that was
converted against `data/assets/smpl/smpl.xml`.

Per-file counts remain:

- joints: 70
- motors: 70
- geoms: 25

Only body offsets and geom dimensions are morphed (plus optional density compensation), so
motion frame width and controller expectations remain aligned with MimicKit's original SMPL rig.

## Regenerate

From repo root:

```bash
python tools/generate_smpl_capsule_variants.py \
  --base_xml data/assets/smpl/smpl.xml \
  --out_dir data/assets/smpl/human_variants_capsule
```

Optional flags:

- `--profiles tall,short,athletic` to output a subset
- `--no_keep_mass` to allow body mass to change with volume
- `--num_random 32 --seed 0` to sample additional random variants
- `--height_min/--height_max` and `--girth_min/--girth_max` to control random diversity range

## Preset scales

The checked-in presets are simple two-parameter morphs of the base rig:

- `neutral`: length `1.000`, girth `1.000`
- `skinny`: length `1.000`, girth `0.850`
- `athletic`: length `1.020`, girth `1.122`
- `stocky`: length `0.970`, girth `1.1155`
- `fat`: length `0.950`, girth `1.235`
- `short`: length `0.920`, girth `0.920`
- `tall`: length `1.080`, girth `1.026`
