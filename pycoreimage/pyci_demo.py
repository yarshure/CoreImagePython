"""
pycoreimage

Copyright 2018 Apple Inc. All rights reserved.

# Install
    1. pip install pyobjc --ignore-installed --user
    2. pip install numpy --ignore-installed --user
    3. pip install scikit-image --user

"""

from pycoreimage.pyci import *


def demo_metadata(fpath):
    img = cimg.fromFile(fpath)
    depth = cimg.fromFile(fpath, useDepth=True)
    matte = cimg.fromFile(fpath, useMatte=True)

    print(img.ciimage.properties())

    # show([img, depth, matte], title=['input', 'depth', 'matte'])
    show(img, at=221, title='RGB {}x{}'.format(*img.size))
    show(depth, at=223, title='depth {}x{}'.format(*depth.size))
    show(matte, at=224, title='matte {}x{}'.format(*matte.size))


def demo_filters(fpath):
    """ Example: filters, loading, saving """

    # Support for most common image file types, including raw.
    img = cimg.fromFile(fpath)

    # Check our image type
    print(type(img))

    # Inspect the backing CIImage
    print(img.ciimage)

    # Print available filters
    for i, f in enumerate(cimg.filters()):
        print('{:3d} {}'.format(i, f))

    # Print more info (including inputs) for a given filter
    print(cimg.inputs('gaussianBlur'))

    radii = [10, 50, 100]
    for i, r in enumerate(radii):
        # Apply a Gaussian blur filter on the input image
        # Note: can also use the full filter name "CIGaussianBlur"
        blur = img.gaussianBlur(radius=r)

        # Save to disk
        blur.save(fpath + '.CIGaussianBlur{}.jpg'.format(r))

        # Display on screen
        show(blur, at='1{}{}'.format(len(radii), i + 1), title='blur with radius={}'.format(r))


def demo_generators():
    """ Example: CoreImage generators. """

    qrcode = cimg.fromGenerator('CIQRCodeGenerator', message='robot barf')
    checker = cimg.fromGenerator('CICheckerboardGenerator', crop=1024)
    stripes = cimg.fromGenerator('CIStripesGenerator', crop=1024)

    show([qrcode, checker, stripes], title=['QR code', 'Checkboard', 'Stripes'], interpolation='nearest')


def demo_numpy_to(fpath):
    """ Example: from CoreImage to NumPy """
    import numpy as np

    # Apply a non trivial effect
    img = cimg.fromFile(fpath)
    vib = img.vibrance(amount=1.0)

    # Render to a NumPy array
    ary = vib.render();

    # Operate on the NumPy array
    print(ary[0, 0, 0])
    print(ary.min())
    coefs = 0.299, 0.587, 0.114
    lum = np.tensordot(ary, coefs, axes=([2, 0]))

    show([img, vib, lum], title=['input', 'vibrance', 'luminance'])


def demo_numpy_from():
    """ Example: from NumPy to CoreImage """
    import numpy as np

    # Create a NumPy array
    noise = np.random.rand(512, 512, 3)
    noise[noise < 0.75] = 0
    show(noise, title='NumPy', interpolation='nearest')

    # CoreImage convenience wrapper
    img = cimg(noise)
    print(type(img))

    # Apply filters
    img = img.discBlur(radius=10).photoEffectChrome()
    img = img.lightTunnel(center=(256, 256), radius=64)
    img = img.exposureAdjust(EV=2)
    img = img.gammaAdjust(power=2)

    show(img, title='NumPy to Core Image')


def demo_slices(fpath):
    """ Example: NumPy-style slicing. """
    import numpy as np

    img = cimg.fromFile(fpath)

    # Resize
    s = img.size
    img = img.resize(1024, preserveAspect=1)
    show(img, title='Resized from {} to {}'.format(s, img.size))

    # Create a blank NumPy array
    labelWidth = 400
    composite = np.zeros((img.size[1], img.size[0] + labelWidth, 3))
    rows = img.size[1] // 5
    show(composite)

    # Create our band processing function
    def addBand(i, name, args):
        # Band indices
        lo, hi = i * rows, (i + 1) * rows

        # Apply filter via explicit name and args
        band = img.applyFilter(name, **args)

        # Create a label with the filter name
        label = cimg.fromGenerator('CITextImageGenerator',
                                   text=name,
                                   fontName='HelveticaNeue',
                                   fontSize=40,
                                   scaleFactor=1)

        # Make the text red
        label = label.colorInvert().whitePointAdjust(color=color(1, 0, 0, 1))

        # Translate to left hand side
        label = label.translate(-labelWidth, composite.shape[0] - hi)

        # Composite over the image
        band = label.over(band)

        # Slice the CIImage here: render only happens in that band
        composite[lo:hi, ...] = band[lo:hi, ...]
        show(composite)

    # Create composite bands using various filters
    addBand(0, 'pixellate', {'center': (0, 0), 'scale': 10})
    addBand(1, 'edgeWork', {'radius': 3})
    addBand(2, 'gaussianBlur', {'radius': 5})
    addBand(3, 'comicEffect', {})
    addBand(4, 'hexagonalPixellate', {'center': (0, 0), 'scale': 10})


def demo_gpu_color(fpath):
    """ Example: GPU color kernel """

    img = cimg.fromFile(fpath)

    # GPU shader written in the Core Image Kernel Language
    # This is a Color Kernel, since only the color fragment is processed
    src = """ 
    kernel vec4 crush_red(__sample img, float a, float b) {
        // Crush shadows from red
        img.rgb *= smoothstep(a, b, img.r);
        return img;
    }
    """
    # Apply
    img2 = img.applyKernel(src,  # kernel source code
                           0.25,  # kernel arg 1
                           0.9)  # kernel arg 2
    show([img, img2], title=['input', 'GPU color kernel'])


def demo_gpu_general(fpath):
    """ Example: GPU general kernel """
    img = cimg.fromFile(fpath)
    img = img.resize(512, preserveAspect=2)

    # Bilateral filter written in the Core Image Kernel Language
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

    # Apply
    sigmaSpatial = 20
    sigmaColor = 0.15
    bil = img.applyKernel(src,  # kernel source
                          3 * sigmaSpatial,  # kernel arg 1
                          sigmaColor ** -2,  # kernel arg 2
                          sigmaSpatial ** -2,  # kernel arg 3
                          # region of interest (ROI) callback
                          roi=lambda index, r: inset(r,
                                                     -3 * sigmaSpatial,
                                                     -3 * sigmaSpatial))

    show([img, bil], title=['input', 'bilateral'])

    # Create the detail layer
    details = img.render() - bil.render()
    show((details - details.min()) / (details.max() - details.min()) ** 1.5, title='detail layer')

    # Bilateral sharpen
    result = img.render() + 1.5 * details
    show([img, result], title=['input', 'bilateral sharpening'])


def demo_geometry(paths):
    """ Example: Affine transformations. """
    import numpy as np

    # Load our images in
    imgs = [cimg.fromFile(path) for path in paths]

    # Composite params
    n = 100
    size = 1024
    composite = None
    np.random.seed(3)

    # Utility randomization function
    def randomize(bar, atCenter=None):
        # Apply random scale
        scale = 0.025 + 0.075 * np.random.rand()

        if atCenter:
            scale = 0.1

        bar = bar.scale(scale, scale)

        # Zero origin
        w, h = bar.size
        bar = bar.translate(-h / 2, -w / 2)

        # Apply random rotation
        angle = np.random.rand() * 2.0 * np.pi
        bar = bar.rotate(angle)

        # Apply random translation
        tx = np.random.rand() * size
        ty = np.random.rand() * size

        if atCenter:
            tx, ty = atCenter

        bar = bar.translate(tx, ty)
        return bar

    # Create the composite
    for i in range(n):
        if composite:
            composite = composite.gaussianBlur(radius=1.5)

        # Pick next image
        bar = imgs[np.random.randint(0, len(imgs))]
        bar = randomize(bar, atCenter=(size / 2, size / 2) if i == n - 1 else None)

        # Composite over the image
        composite = bar if not composite else bar.over(composite)

    # Crop to input size
    composite = composite.crop(size, size)

    show(composite)


def demo_depth(fpath):
    """ Example: depth processing """
    import numpy as np

    # Load image
    foo = cimg.fromFile(fpath)
    W, H = foo.size

    # Load depth
    depth = cimg.fromFile(fpath, useDepth=True)
    w, h = depth.size

    # Diagonal
    d = np.sqrt(w ** 2 + h ** 2)

    # Params
    blur = min(100, 0.2 * d)
    morpho = blur / 6.0
    perc = 20

    # Threshold depth at 75th percentile
    depth_img = depth.render()
    p = np.percentile(depth_img, perc)
    mask = depth_img < p
    mask = cimg(mask.astype(np.float32))
    mask = mask.morphologyMaximum(radius=morpho)
    mask = mask.gaussianBlur(radius=blur)
    mask = mask.render()

    # Downscale original
    ds = cimg(foo).scale(w / float(W), h / float(H))

    # Desaturate the background
    bg = ds.photoEffectNoir()

    # Make the foreground stand out
    fg = ds.exposureAdjust(EV=0.5).colorControls(saturation=0.8, contrast=1.4)

    # Render
    bg = bg.render()
    fg = fg.render()

    # Make the foreground stand out
    result = mask * fg + (1 - mask) * bg

    # Show on screen
    show(ds, at=221, title='input')
    show(depth, at=222, color='jet', title='depth')
    show(result, at=223, title='result')
    show(mask[..., 0], at=224, color='gray', title='mask')


def demo_depth_blur(fpath):
    """ Example: depth blur"""
    img = cimg.fromFile(fpath)
    matte = cimg.fromFile(fpath, useMatte=True)
    disparity = cimg.fromFile(fpath, useDisparity=True)

    for aperture in [2, 6, 22]:
        effect = img.depthBlurEffect(
            inputDisparityImage=disparity,
            inputMatteImage=matte,
            inputAperture=aperture
        )
        show(effect, title='Aperture = {}'.format(aperture))


if __name__ == '__main__':

    import argparse, os

    # Support file formats for dataset demo
    exts = '.jpg', '.jpeg', '.heic', '.tiff', '.png'

    # print('Syntax: pycoreimage_sandbox.py img.jpg imgDepthMatte.heic /path/to/directory/')
    parser = argparse.ArgumentParser()

    parser.add_argument('image', help='input image', type=str)
    parser.add_argument('directory', help='directory containing images {}'.format(exts), type=str)
    parser.add_argument('--imageWithDepth', help='input image containing depth and Portrait Effects Matte', type=str)
    parser.add_argument('--tree', action='store_true', help='enable CI_PRINT_TREE=4')
    args = parser.parse_args()

    # CI_PRINT_TREE needs to be set before the first render call
    if args.tree:
        set_print_tree(4)

    # Input check
    abort = False
    if not os.path.exists(args.image):
        abort = True
        print('Image not found:', args.image)

    if args.imageWithDepth:
        if not os.path.exists(args.imageWithDepth):
            abort = True
            print('Depth+matte image not found:', args.imageWithDepth)

    if not os.path.exists(args.directory):
        abort = True
        print('Directory not found:', args.directory)

    # Only process selected bitmap file formats
    dataset = []
    if not abort:
        dataset = os.listdir(args.directory)
        dataset = [os.path.join(args.directory, f) for f in dataset if os.path.splitext(f)[1].lower() in exts]

    if len(dataset) == 0:
        abort = True
        print('No valid bitmap ({}) found in:'.format(exts), args.directory)

    if abort:
        exit(1)

    print('Using input image', args.image)

    if args.imageWithDepth:
        print('Using input image with depth', args.imageWithDepth)
    else:
        print('Not using an image with depth. Set --imageWithDepth')

    print('Using dataset', dataset)

    # Demos

    print('Running demo: filters')
    demo_filters(args.image)

    print('Running demo: generators')
    demo_generators()

    print('Running demo: to numpy')
    demo_numpy_to(args.image)

    print('Running demo: from numpy')
    demo_numpy_from()

    print('Running demo: slicing')
    demo_slices(args.image)

    print('Running demo: GPU kernels (color)')
    demo_gpu_color(args.image)

    print('Running demo: GPU kernels (general)')
    demo_gpu_general(args.image)

    print('Running demo: geometry dataset')
    demo_geometry(dataset)

    if args.imageWithDepth:
        try:
            print('Running demo: depth processing')
            demo_depth(args.imageWithDepth)

            print('Running demo: depth blur')
            demo_depth_blur(args.imageWithDepth)

        except Exception as e:
            print('Encountered an exception while preparing the Portrait depth and matte demo', e)
            print('Make sure your input image ({}) has both embedded as metadata.'.format(args.imageWithDepth))
