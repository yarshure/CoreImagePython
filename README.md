# Prototyping Your App's Core Image Pipeline with Python

Write and test Core Image filters, kernels, and geometric transformations in Python.

## Overview

The Python bindings available in `pycoreimage` allow you to prototype your app’s Core Image pipeline without writing production code in Objective-C or Swift, accelerating the process of fine-tuning filters and experimenting with Core Image effects. This sample code project demonstrates `pycoreimage` filter usage for common image processing operations, such as depth filtering, color warping, and geometric transformation.

## Install `pycoreimage`

The version of Python that shipped with your system may not correspond to the most recent release of Python; after upgrading to Python 3.5, set up your Python environment by downloading and installing `pip` beforehand by inputting the command into Terminal:

    >>> easy_install pip

`pip` is an installer that you'll use to fetch and install Python packages, such as `numpy` for numerical computing and `scikit-image` for displaying images.  Ensure `setuptools` is up to date by calling the command:

    >>> pip install setuptools --upgrade

Next, install `pycoreimage` by executing the following script from your Terminal command line:

    >>> python setup.py --user

The script installs necessary packages to run the demo, such as `numpy` and `pyobjc`, so you don't need to install them separately.

The console will ask to install `pycoreimage`.  This allows your program to access Core Image functionality such as Python bindings for Core Image filters. From Python, you can apply the Portrait Matte effect, custom GPU kernels, barcode generators, and geometrical transformations.

The package `scikit-image` is not required for `pycoreimage`, but it is necessary to run the sample demo since we are displaying images on the screen.

    >>> pip install scikit-image --user

## Run the Sample Code in a Python Environment

The sample code loads and runs in Xcode, but you can also run it in your preferred Python interpreter.  Editing the code in an interpreter, you can see the resulting image change in real-time as you might in a REPL compiler or IDE.

## Load Images Using `pycoreimage`

`pycoreimage` supports much of the functionality that Core Image offers to replicate the recipe-based environment of image processing on GPUs.

Start by importing the Core Image class from `pyci`:

```
from pycoreimage.pyci import cimg
```

`pycoreimage` supports all file formats that Core Image supports, including JPEG, PNG, and HEIC.  

Load images of any standard file type from disk into the native cimg format.

```
fpath = 'resources/YourFacialImage.HEIC'
image = cimg.fromFile(fpath)
```

## Work with Depth Data

Obtain the image size using the `size` property:

```
W, H = image.size
```

Access depth data that you can use to perform image processing operations such as thresholding, segmentation, and mask morphology:

```
depth = cimg.fromFile(fpath, useDepth=True)
w, h = depth.size

# Set the threshold depth at the 50th percentile.
depth_img = depth.render()[..., :3]
p = np.percentile(depth_img, 50)
mask = depth_img < p
mask = cimg(mask.astype(np.float32))
mask = mask.morphologyMaximum(radius=5)
mask = mask.gaussianBlur(radius=30)
mask = mask.render()
```

Render the final result in the Python interpreter, with immediate feedback in real time:

```
show(img.clip(0, 1), 221)
show(depth.render()[..., 0].clip(0, 1), 222, map='jet')
show(img_feather.clip(0, 1), 223, suptitle='Demo 6: depth')
show(mask[..., 0], 224, map='gray')
```

## See the Portrait Matte Effect

The demo in `pyci_demo.py` applies Core Image filters to achieve a number of image processing effects, as shown in the [WWDC presentation](https://developer.apple.com/videos/play/wwdc2018/719/), but you must substitute your own facial images with the corresponding portrait effect depth data to see the Portrait Matte effect.

Using a facial image captured in portrait mode on iOS 12, you can load the Portrait Matte effect with the following command:

```
matte = cimg.fromFile(fpath, useMatte=True)
```

## Apply Core Image Filters by Name

With your image loaded, you can apply any of over 200 Core Image filters, calling the same methods that Objective-C calls in a production environment.  See [Core Image Filter Reference](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Reference/CoreImageFilterReference/) for a listing of these filters.  You can invoke any filter explicitly by name:

```
img = img.CIGaussianBlur(radius=1)
```

Alternatively, you can invoke and apply filters with their string names by using the `applyFilter` function:

```
# Load an input image from file.
img = cimg.fromFile('resources/sunset_1.png')

# Resize to half size.
img = img.scale(0.5, 0.5)

# Create a blank image.
composite = np.zeros((img.size[1], img.size[0], 3))

filters = 'pixellate', 'edgeWork', 'gaussianBlur', 'comicEffect', 'hexagonalPixellate'
rows = int(img.size[1]) / len(filters)
for i, filter in enumerate(filters):
    # Apply the filter.
    slice = img.applyFilter(filter)

    # Slice and add to composite.
    lo = i * rows
    hi = (i + 1) * rows
    composite[lo:hi, :, :3] = slice[lo:hi, :, :3]
```

You can query available filters by their name by using the `print` command:

```
print(cimg.filters())
```

For a given filter, query its available outputs by using the `inputs` property:

```
print(cimg.inputs('gaussianBlur'))
```


## Define and Apply a Custom GPU Kernel to an Image

In the Python prototyping environment, you can create a custom kernel by writing an inline shader in the Core Image Kernel Language. For example, write a color kernel by processing only the color fragment. The `src` keyword tells Python that the code enclosed in `"""` is kernel source code:

```
src = """
kernel vec4 crush_red(__sample img, float a, float b) {
    // Crush shadows from red.
    img.rgb *= smoothstep(a, b, img.r);
    return img;
}
"""
```

Apply the GPU kernel to an image by calling `applyKernel`:

```
img2 = img.applyKernel(src,  # kernel source code
                       0.25,  # kernel arg 1
                       0.9)  # kernel arg 2
show([img, img2], title=['input', 'GPU color kernel'])
```

For instance, apply a general bilateral filter with the following kernel, written in the Core Image Kernel Language:

```
src = """
kernel vec4 bilateral(sampler u, float k, float colorInv, float spatialInv)
{
  vec2 dc = destCoord();
  vec2 pu = samplerCoord(u);
  vec2 uDelta = samplerTransform(u, dc+vec2(1.0)) - pu;
  vec4 u_0 = sample(u, pu);
  vec4 C = vec4(0.0);
  float W = 0.0;
  for (float x = -k; x <= k; x++) {
    for (float y = -k; y <= k; y++){
      float ws = exp(-(x*x+y*y) * spatialInv);
      vec4 u_xy  = sample(u, pu + vec2(x,y)*uDelta);
      vec3 diff = u_xy.rgb-u_0.rgb;
      float wc = exp(-dot(diff,diff) * colorInv);
      W += ws * wc;
      C += ws * wc * u_xy;
    }
  }
  return W < 0.0001 ? u_0 : C / W;
}
"""
```

## Generate a QR Code from Data

A number of `CIFilter`s, like all barcode-creating filters, generate procedural images and don't take input images.  For example, you can create a QR code from arbitrary text with the `fromGenerator` function:

```
cimg.fromGenerator('CIQRCodeGenerator', message='Hello World!')
```

## Apply Geometrical Transformations

Shift the image by the amount `(tx, ty)` with the `translate` command:

```
img.translate(tx, ty)
```

Use the `scale` command to resize an image:

```
img.scale(sx, sy)
```

Rotate the image about its center point with the `rotate` command:

```
img.rotate(radians)
```

Crop the image to the rectangle `[x, y, width, height]` with the `crop` command:

```
img.crop(0, 0, 1024, 768)
```

## Fine-Tune Your Output Image Live

One advantage to using Python to prototype a filter chain is its immediate feedback.  Write out the resulting image with the `save` command:

```
img.save('demo2.jpg')
```

Calling `show` displays the image onscreen in the Python editor:

```
show(img, title='Demo 2: from file + slicing')
```

The sample code contains a number of other common image-processing routines you can customize for your app, such as sharpening kernels, zoom and motion blur, and image slicing.  Create a set of test images and run the code on them to see the effects live.
