# Offline comparison

Open [index.html](index.html) in a browser. Select PCA, LDA, LPP or MFA-1, then click test points to inspect their digit images and six nearest training neighbors.

Data: fixed Digits classes 0/1/2, 179 training and 162 test images, shared train-only PCA98, orthonormal 2D subspaces. Colors and symbols represent true classes; crosses mark errors. Animation interpolates display positions, not training dynamics.

The fixed query is image 56. Image 51 is a post-fitting disagreement example and was not used for tuning. Coordinates, predictions and thumbnails are embedded, so no server or network is needed.

To rebuild from the saved comparison:

```sh
python demo/build_demo.py
```

To recompute the comparison first, run `python code/visual_comparison.py`. This 2D teaching comparison is separate from the saved full classification study.
